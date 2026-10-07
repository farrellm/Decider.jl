# Decider.jl Design

Julia front end for decision models (TypeSafe Jev, OpenRouter decision models, OpenAI Decisions API). API facts verified against provider docs on 2026-10-07.

## 1. Goals / non-goals
- **Goals:** one Julia API for asking a decision model a typed question. Results come back as either (a) a map from the user's own option type (an `@enum` instance or a `Symbol`) to a probability, or (b) the single most likely value. Supported backends: the TypeSafe Jev API, Jev on OpenRouter, any other OpenRouter decision model (clef-flash, clef, d1, pplx-decider, mercury-decide, solar-decide, …), and the OpenAI Decisions API.
- **Non-goals (v1):** text-generation LLMs, streaming, local model inference, building prompts for the user, client-side calibration.

## 2. Backend landscape (what we are wrapping)

There are two wire formats:

| | TypeSafe System One (TypeSafe, OpenRouter) | OpenAI Decisions |
|---|---|---|
| Endpoint | `POST https://api.typesafe.ai/v1/systemone`; OpenRouter: `POST https://openrouter.ai/api/v1/systemone` or `POST https://openrouter.ai/api/alpha/decisions` (same schema, plus `provider`, `session_id`, `user`, `trace`) | `POST https://api.openai.com/v1/decisions` |
| Auth | `Authorization: Bearer` (`TYPESAFE_API_KEY` / `OPENROUTER_API_KEY`) | `Authorization: Bearer` (`OPENAI_API_KEY`) |
| Input | `state`: string \| object \| array (text only for Jev) | `input`: string, or user messages with `input_text` and `input_image` parts (base64 data URLs only) |
| Questions | `questions`: **map** id → question | `questions`: **array** of questions with a `name` field; max 10 per request |
| Yes/no | `type:"noul"`, `criteria:{true,false}` (optional) → `{"noul": p}` | `type:"predicate"` → `{"probability": p}` |
| Choice | `criteria`: map option → description\|null (≤255) → `choice`, `probabilities` map, `confidence` | `choices:[{value,description}]` → `choice`, `probabilities:[{value,probability}]`, `confidence` |
| Score | `criteria`: ordered array of level descriptions (2–10) → `score`, `legend`, `probabilities` {"0":p,…}, `confidence` | `levels:[{label,description}]` → `score`, `probabilities:[{value,label,probability}]`, `confidence` |
| Response extras | `model`, `usage{input_tokens,output_tokens[,cost]}`, OpenRouter `id`, `provider` | `answers` array |
| Errors | 401, 402 (OpenRouter), 422/400, 429, 529 | standard OpenAI errors |

Models known on OpenRouter (from `output_modalities=decisions`): `typesafe/jev-1.13`, `~typesafe/jev-latest`, `cloudflare/clef-flash`, `cloudflare/clef`, `liquid/d1`, `perplexity/pplx-decider-v1[.1]-27b`, `inception/mercury-decide:free`, `upstage/solar-decide`, `togethercomputer/tev1-4b-experimental` (choice only, 2–24 options), `respan/span-01[-lite]` (behavior probabilities), `jaredpalmer/kev-4b`, `openai/gpt-6-luna-decisions`. Capabilities differ by model. In OpenRouter's schema, choice/score `probabilities` and `confidence` are **optional**, so some models may return only the chosen value.

**Consequence for the design:** we need one internal question model, two wire codecs (SystemOne and OpenAI), and three user-facing backend constructors.

## 3. Public API

```julia
using Decider

@enum Dept billing technical sales
jev = TypeSafe()                                 # key from ENV["TYPESAFE_API_KEY"], model "jev-latest"
q   = Choice(Dept, "Which team should handle this?")

decide(jev, q, "I was charged twice.")           # -> billing::Dept
probabilities(jev, q, "I was charged twice.")    # -> OrderedDict(billing=>0.88, technical=>0.12, sales=>0.0)
a = ask(jev, q, "I was charged twice.")          # -> Answer{Dept}: .value .probabilities .confidence .model .usage .raw
```

Three verbs with the same signature `(backend, question(s), state; kw...)`. The backend argument is always required; there is no global or task-scoped default backend:
- `ask` returns the full `Answer{T}`.
- `probabilities` returns `OrderedDict{T,Float64}`, in the order the options were declared.
- `decide` returns the argmax `T`.

**Questions** (`AbstractQuestion{T}`, where `T` is the result key type):
- `Choice(E::Type{<:Enum}, instructions; descriptions=nothing)`, giving `T = E`.
- `Choice([:a, :b, :c], instructions)` or `Choice([:a => "desc", :b => "desc"], instructions)`, giving `T = Symbol`.
- `YesNo(instructions; yes=nothing, no=nothing)`, giving `T = Bool`. `probabilities` returns `Dict{Bool,Float64}(true=>p, false=>1-p)` (decided: uniform with Choice/Score, not a bare Float64) and `decide` returns `p ≥ 0.5`. Maps to `noul` / `predicate`. A `threshold` kwarg on `decide` is accepted.
- `Score(levels, instructions)`, where levels is an Enum type (declaration order = ordinal), a `Vector{Symbol}`, or `Vector{Pair{Symbol,String}}`. `T` is the level type. `Answer` also carries `.score::Float64`, the expected ordinal. `decide` returns the most likely level. Docs should mention that `round(score)` is often the better choice.
- `instructions` can be a `String` or any JSON-serializable value (Dict, NamedTuple), matching TypeSafe's structured instructions.
- Descriptions for Enum options come from the `descriptions` kwarg (Dict/pairs) or from an overridable hook `Decider.describe(x::MyEnum)::Union{Nothing,String}`, which defaults to `nothing`.
- Option wire strings come from `string(x)`. A `Dict{String,T}` inverse is built at construction. The constructor validates uniqueness, non-emptiness, ≤255 choices, and 2–10 score levels.

**Multiple questions in one request:** pass a `NamedTuple` of questions. The result is a `NamedTuple` with the same keys:
```julia
decide(jev, (dept=Choice(Dept, "..."), urgent=YesNo("Is this urgent?")), ticket)
# -> (dept = billing, urgent = true)
```
NamedTuple keys become question ids/names. For OpenAI the request is split into chunks of ≤10 questions, sent concurrently, and the results merged. A single question uses the internal id `"q"`.

**State:** `AbstractString`, or anything JSON3 can serialize (Dict, NamedTuple, Vector, StructTypes structs). `Image(bytes, mime)` / `Image(path)` values may appear in a `Vector` state. Only backends that declare image support accept them; others throw `ArgumentError` before sending.

## 4. Backends

```julia
abstract type AbstractBackend end

TypeSafe(; api_key=ENV["TYPESAFE_API_KEY"], model="jev-latest",
           base_url="https://api.typesafe.ai")                       # SystemOne codec
OpenRouter(; api_key=ENV["OPENROUTER_API_KEY"], model="typesafe/jev-1.13",
             endpoint=:decisions,   # :decisions => /api/alpha/decisions, :systemone => /api/v1/systemone
             provider=nothing, session_id=nothing, user=nothing)      # SystemOne codec
OpenAIDecisions(; api_key=ENV["OPENAI_API_KEY"], model="gpt-6-luna",
                  base_url="https://api.openai.com")                  # OpenAI codec
```
- All backends also take `timeout`, `retries` (default 3), and `transport` (see Testing). API keys are read lazily, so constructing a backend without a key set does not throw until a request is made.
- Internally: `codec(::AbstractBackend)` returns `SystemOneCodec()` or `OpenAICodec()`. Each codec implements `encode(codec, questions, state)::Dict` and `decode(codec, questions, json)::NamedTuple{…,Answer}`. Backends implement `endpoint_url`, `headers`, `max_questions` (OpenAI 10, others `typemax(Int)`), and `supports_images` (OpenAI true, TypeSafe false, OpenRouter: see open questions).
- Adding a new provider that speaks either wire format means defining one struct and four small methods.

## 5. Decoding and normalization rules
- Map wire option strings back to `T` through the inverse dict. An unknown option string from the server raises `DecoderError`.
- If an option is missing from the returned probabilities, fill it with 0.0. Then renormalize if `|sum − 1| > 1e-6`, and warn when the deviation is above 1e-3.
- If `probabilities` is absent (allowed by OpenRouter's schema for some models), `decide` still works using `choice`. `probabilities`/`ask` throw `ProbabilitiesUnavailable` unless called with `allow_point_mass=true`, which returns a one-hot map.
- Score: SystemOne returns probability keys `"0".."n-1"` and OpenAI returns `value` indices. Both map to levels by ordinal position.
- Noul/predicate: `p` is clamped to [0,1].
- `Answer` keeps `raw` (the parsed JSON for that question), plus `model` and `usage` from the response.

## 6. Errors, retries, HTTP
- `DeciderAPIError(status, message, body)` for non-2xx responses. Retry 429, 529, 5xx, and connection errors with exponential backoff and jitter, honoring `retry-after`. Do not retry 400/401/402/422.
- Dependencies: `HTTP.jl`, `JSON3.jl`, `OrderedCollections.jl`, and the `Base64` stdlib. No StructTypes-based schema types: requests are built as `Dict`s, which keeps the codecs short.
- Thread/task safety: backends are immutable, and concurrent calls (`asyncmap` over many states) are the supported way to batch across inputs.

## 7. Source layout (for later implementation)
```
src/Decider.jl          # module, exports
src/questions.jl        # AbstractQuestion, Choice, YesNo, Score, describe, validation
src/answers.jl          # Answer{T}, ask/probabilities/decide front ends, NamedTuple fan-out
src/state.jl            # Image, state validation
src/backends.jl         # AbstractBackend, TypeSafe, OpenRouter, OpenAIDecisions
src/codecs/systemone.jl
src/codecs/openai.jl
src/http.jl             # request, retry/backoff, errors
```
Exports: `ask, decide, probabilities, Choice, YesNo, Score, Image, Answer, TypeSafe, OpenRouter, OpenAIDecisions, DeciderAPIError`.

## 8. Testing strategy
- `transport` is a function `(method, url, headers, body) -> (status, headers, body)`. It defaults to an HTTP.jl call, and tests inject canned responses.
- Golden tests: for each codec, compare the encoded request JSON to the doc examples. Decode the example responses from TypeSafe, OpenRouter, and OpenAI in §2 (taken verbatim from the docs) into `Answer`s.
- Unit tests: Enum/Symbol round-trip, renormalization, missing probabilities, OpenAI chunking at 10, retry on 429/529, image rejection on text-only backends, validation errors.
- Live tests run only when `DECIDER_LIVE_TESTS=1` and the relevant API key is set. Each is one cheap request per backend (OpenRouter with `jev-1.13` and `clef-flash`).
- Aqua stays on.

## 9. Open questions
1. Does OpenRouter `/api/alpha/decisions` accept images in `state` for image-capable models (clef, gpt-6-luna-decisions)? The schema only shows text/JSON. v1 assumes it does not.
2. Default OpenRouter endpoint: the `alpha/decisions` router covers all decision models but is alpha. `/v1/systemone` is stable but may be limited to System One-namespace models.
