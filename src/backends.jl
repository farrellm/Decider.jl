"""
    AbstractBackend

A decision-model provider. To add one that speaks an existing wire format, define a struct
with fields `model`, `retries`, and `transport`, and the methods [`codec`](@ref),
[`endpoint_url`](@ref), and [`headers`](@ref); override [`max_questions`](@ref),
[`supports_images`](@ref), and [`request_extras`](@ref) when the defaults do not fit.
"""
abstract type AbstractBackend end

struct SystemOneCodec end
struct OpenAICodec end

"""
    codec(backend)

The wire format the backend speaks: `SystemOneCodec()` or `OpenAICodec()`.
"""
function codec end

"""
    endpoint_url(backend)::String

The URL decision requests are POSTed to.
"""
function endpoint_url end

"""
    headers(backend)::Vector{Pair{String,String}}

Request headers, including authorization. Called once per request, so API keys are read
from the environment lazily.
"""
function headers end

"""
    max_questions(backend)::Int

Most questions the backend accepts in one request; larger batches are split and sent
concurrently. Defaults to `typemax(Int)`.
"""
max_questions(::AbstractBackend) = typemax(Int)

"""
    supports_images(backend)::Bool

Whether [`Image`](@ref) values may appear in the state. Defaults to `false`.
"""
supports_images(::AbstractBackend) = false

"""
    request_extras(backend)::Dict{String,Any}

Extra top-level fields merged into every request body. Defaults to none.
"""
request_extras(::AbstractBackend) = Dict{String,Any}()

"""
    HTTPTransport(timeout)

The default transport: POSTs with HTTP.jl and returns `(status, headers, body)` without
throwing on non-2xx statuses. Retries are handled by Decider, not HTTP.jl.
"""
struct HTTPTransport
    timeout::Float64
end

function (t::HTTPTransport)(method, url, headers, body)
    resp = HTTP.request(method, url, headers, body;
        status_exception = false, retry = false, redirect = false,
        request_timeout = t.timeout)
    return resp.status, Pair{String,String}[k => v for (k, v) in resp.headers],
    String(resp.body)
end

_transport(transport, timeout) = transport === nothing ? HTTPTransport(timeout) : transport

function _api_key(key, var)
    key !== nothing && return key
    value = get(ENV, var, "")
    isempty(value) && throw(ArgumentError("no API key: pass api_key or set ENV[\"$var\"]"))
    return value
end

_bearer(key) = ["Authorization" => "Bearer $key", "Content-Type" => "application/json"]

function Base.show(io::IO, b::AbstractBackend)
    return print(io, nameof(typeof(b)), "(model = ", repr(b.model), ")")
end

"""
    TypeSafe(; api_key=ENV["TYPESAFE_API_KEY"], model="jev-latest",
               base_url="https://api.typesafe.ai", timeout=60, retries=3, transport=nothing)

The TypeSafe System One API (`POST /v1/systemone`). Text and JSON state only. The API key
is read when a request is made, not at construction.

`transport` replaces the HTTP call; it is a function
`(method, url, headers, body) -> (status, headers, body)`.
"""
struct TypeSafe <: AbstractBackend
    api_key::Union{Nothing,String}
    model::String
    base_url::String
    retries::Int
    transport::Any
end

function TypeSafe(; api_key = nothing, model = "jev-latest",
    base_url = "https://api.typesafe.ai", timeout = 60, retries = 3, transport = nothing)
    return TypeSafe(api_key, model, rstrip(base_url, '/'), retries,
        _transport(transport, timeout))
end

codec(::TypeSafe) = SystemOneCodec()
endpoint_url(b::TypeSafe) = b.base_url * "/v1/systemone"
headers(b::TypeSafe) = _bearer(_api_key(b.api_key, "TYPESAFE_API_KEY"))

const OPENROUTER_ENDPOINTS = Dict(
    :decisions => "/api/alpha/decisions",
    :systemone => "/api/v1/systemone",
)

"""
    OpenRouter(; api_key=ENV["OPENROUTER_API_KEY"], model="typesafe/jev-1.13",
                 endpoint=:decisions, provider=nothing, session_id=nothing, user=nothing,
                 base_url="https://openrouter.ai", timeout=60, retries=3, transport=nothing)

Any OpenRouter decision model, over the System One wire format. `endpoint=:decisions`
uses `/api/alpha/decisions`, which routes to every decision model and accepts the
`provider` (routing preferences, any JSON-serializable value), `session_id`, and `user`
fields; `endpoint=:systemone` uses the stable `/api/v1/systemone`.

Images in the state are not supported.
"""
struct OpenRouter <: AbstractBackend
    api_key::Union{Nothing,String}
    model::String
    url::String
    extras::Dict{String,Any}
    retries::Int
    transport::Any
end

function OpenRouter(; api_key = nothing, model = "typesafe/jev-1.13",
    endpoint::Symbol = :decisions, provider = nothing, session_id = nothing,
    user = nothing, base_url = "https://openrouter.ai", timeout = 60, retries = 3,
    transport = nothing)
    path = get(OPENROUTER_ENDPOINTS, endpoint, nothing)
    path === nothing && throw(ArgumentError(
        "endpoint must be :decisions or :systemone, got :$endpoint"))
    extras = Dict{String,Any}()
    for (k, v) in (("provider", provider), ("session_id", session_id), ("user", user))
        v === nothing && continue
        endpoint == :decisions || throw(ArgumentError("$k requires endpoint = :decisions"))
        extras[k] = v
    end
    return OpenRouter(api_key, model, rstrip(base_url, '/') * path, extras, retries,
        _transport(transport, timeout))
end

codec(::OpenRouter) = SystemOneCodec()
endpoint_url(b::OpenRouter) = b.url
headers(b::OpenRouter) = _bearer(_api_key(b.api_key, "OPENROUTER_API_KEY"))
request_extras(b::OpenRouter) = b.extras

"""
    OpenAIDecisions(; api_key=ENV["OPENAI_API_KEY"], model="gpt-6-luna",
                      base_url="https://api.openai.com", timeout=60, retries=3,
                      transport=nothing)

The OpenAI Decisions API (`POST /v1/decisions`). Accepts [`Image`](@ref)s in a `Vector`
state. Requests with more than 10 questions are split and sent concurrently.
"""
struct OpenAIDecisions <: AbstractBackend
    api_key::Union{Nothing,String}
    model::String
    base_url::String
    retries::Int
    transport::Any
end

function OpenAIDecisions(; api_key = nothing, model = "gpt-6-luna",
    base_url = "https://api.openai.com", timeout = 60, retries = 3, transport = nothing)
    return OpenAIDecisions(api_key, model, rstrip(base_url, '/'), retries,
        _transport(transport, timeout))
end

codec(::OpenAIDecisions) = OpenAICodec()
endpoint_url(b::OpenAIDecisions) = b.base_url * "/v1/decisions"
headers(b::OpenAIDecisions) = _bearer(_api_key(b.api_key, "OPENAI_API_KEY"))
max_questions(::OpenAIDecisions) = 10
supports_images(::OpenAIDecisions) = true
