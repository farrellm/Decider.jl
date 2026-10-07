# Decider

[![Build Status](https://github.com/farrellm/Decider.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/farrellm/Decider.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/farrellm/Decider.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/farrellm/Decider.jl)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://farrellm.github.io/Decider.jl/dev/)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Julia front end for decision models: models that answer typed questions about some state
with calibrated probabilities instead of text. Results come back keyed by your own option
type, either an `@enum` or `Symbol`s.

Supported backends:

- `TypeSafe()`: the TypeSafe System One API (Jev). Key from `TYPESAFE_API_KEY`.
- `OpenRouter(; model)`: any OpenRouter decision model (`typesafe/jev-1.13`,
  `cloudflare/clef-flash`, `liquid/d1`, …). Key from `OPENROUTER_API_KEY`.
- `OpenAIDecisions()`: the OpenAI Decisions API. Key from `OPENAI_API_KEY`.

API keys are read when a request is made, so constructing a backend never fails for a
missing key.

## Usage

```julia
using Decider

@enum Dept billing technical sales
jev = TypeSafe()
q = Choice(Dept, "Which team should handle this?")

decide(jev, q, "I was charged twice.")          # billing
probabilities(jev, q, "I was charged twice.")   # OrderedDict(billing => 0.88, technical => 0.12, sales => 0.0)
a = ask(jev, q, "I was charged twice.")         # Answer{Dept}: .value .probabilities .confidence .model .usage .raw
```

All three verbs take `(backend, question(s), state; kw...)`:

- `ask` returns the full `Answer{T}`.
- `probabilities` returns an `OrderedDict{T,Float64}` in declaration order.
- `decide` returns the most likely `T`.

### Questions

```julia
Choice(Dept, "Which team?"; descriptions = Dict(billing => "Charges, refunds, invoices"))
Choice([:refund, :exchange, :other], "What does the customer want?")
Choice([:spam => "Unsolicited advertising", :ham], "Is this spam?")

YesNo("Is this urgent?"; yes = "Needs action today")  # T = Bool
Score([:low, :medium, :high], "How frustrated is the customer?")
```

- `YesNo`: `probabilities` gives `true => p, false => 1 - p`, and
  `decide(...; threshold = 0.5)` gives `p ≥ threshold`.
- `Score`: `Answer` also has `.score`, the expected 0-based ordinal. `decide` returns the
  single most likely level. To pick a level that reflects the whole distribution,
  `levels[round(Int, a.score) + 1]` is often the better choice.
- Instructions may be a string or any JSON-serializable value.
- Enum options can get their descriptions from a method:
  `Decider.describe(x::Dept) = ...`.

### Several questions in one request

```julia
decide(jev, (dept = Choice(Dept, "Which team?"), urgent = YesNo("Is this urgent?")), ticket)
# (dept = billing, urgent = true)
```

OpenAI accepts at most 10 questions per request. Larger sets are split into several
requests, sent concurrently, and the results merged.

### State

The state can be a string or anything JSON3 can serialize. With `OpenAIDecisions`, a
`Vector` state may also contain images:

```julia
decide(OpenAIDecisions(), YesNo("Is there a cat?"), ["Photo:", Image("cat.png")])
```

To batch over many inputs, run calls concurrently. Backends are immutable and safe to
share:

```julia
asyncmap(t -> decide(jev, q, t), tickets)
```

### When probabilities are missing

Some OpenRouter models return only their chosen value. `decide` still works. `ask` and
`probabilities` throw `Decider.ProbabilitiesUnavailable` unless you pass
`allow_point_mass = true`, which returns a one-hot distribution.

### Errors and retries

A non-2xx response throws `DeciderAPIError(status, message, body)`. Statuses 429, 529, and
5xx, and connection errors, are retried with exponential backoff and jitter, honoring
`Retry-After` (`retries = 3` by default). A response that cannot be mapped onto the
question, such as an unknown option or a refusal, throws `Decider.DecoderError`.

## Testing

`Pkg.test()` uses canned responses. To also make one cheap request to each backend whose
API key is set, run with `DECIDER_LIVE_TESTS=1`.
