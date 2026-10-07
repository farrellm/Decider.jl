"""
    DeciderAPIError(status, message, body)

The provider answered with a non-2xx status. `message` is extracted from the error body
when possible; `body` is the raw response text.
"""
struct DeciderAPIError <: Exception
    status::Int
    message::String
    body::String
end

function Base.showerror(io::IO, e::DeciderAPIError)
    return print(io, "DeciderAPIError: HTTP ", e.status, ": ", e.message)
end

"""
    DecoderError(message)

The provider's response could not be mapped onto the question: a missing answer, an
unknown option, or a refusal.
"""
struct DecoderError <: Exception
    message::String
end

Base.showerror(io::IO, e::DecoderError) = print(io, "DecoderError: ", e.message)

const RETRY_STATUSES = (429, 529)
_retryable(status) = status in RETRY_STATUSES || 500 <= status <= 599

_retryable(e::Exception) =
    e isa Union{Base.IOError,EOFError,HTTP.ConnectError,
        HTTP.TimeoutError,HTTP.ProtocolError,HTTP.TLSTransportError}

# Overridden in tests so retries do not sleep.
const SLEEP = Ref{Any}(sleep)

const MAX_BACKOFF = 30.0

function _retry_after(hdrs)
    for (k, v) in hdrs
        lowercase(k) == "retry-after" || continue
        s = tryparse(Float64, strip(v))
        return s === nothing ? nothing : clamp(s, 0.0, 60.0)
    end
    return nothing
end

_backoff(attempt) = min(MAX_BACKOFF, 0.5 * 2.0^(attempt - 1)) * (0.5 + rand())

function _error_message(status, body)
    json = try
        JSON3.read(body)
    catch
        nothing
    end
    if json isa JSON3.Object
        err = get(json, :error, nothing)
        err isa JSON3.Object && (err = get(err, :message, nothing))
        err isa AbstractString && return String(err)
        for k in (:message, :detail)
            m = get(json, k, nothing)
            m === nothing || return m isa AbstractString ? String(m) : JSON3.write(m)
        end
    end
    isempty(body) || return first(body, 500)
    return "HTTP $status"
end

# POST `body` to the backend, retrying transient failures, and return the parsed JSON.
function post(backend::AbstractBackend, body)
    url = endpoint_url(backend)
    hdrs = headers(backend)
    payload = JSON3.write(body)
    attempt = 0
    while true
        result = try
            backend.transport("POST", url, hdrs, payload)
        catch e
            (_retryable(e) && attempt < backend.retries) || rethrow()
            attempt += 1
            SLEEP[](_backoff(attempt))
            continue
        end
        status, rhdrs, rbody = result
        if 200 <= status <= 299
            return try
                JSON3.read(rbody)
            catch e
                e isa ArgumentError || rethrow()
                throw(DecoderError("response is not valid JSON: $(first(rbody, 200))"))
            end
        end
        if _retryable(status) && attempt < backend.retries
            attempt += 1
            SLEEP[](something(_retry_after(rhdrs), _backoff(attempt)))
            continue
        end
        throw(DeciderAPIError(status, _error_message(status, rbody), rbody))
    end
end
