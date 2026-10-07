"""
    Answer{T}

A decision model's answer to one question.

- `value::T`: the most likely option (for [`YesNo`](@ref), `p ≥ 0.5`).
- `probabilities::OrderedDict{T,Float64}`: the distribution, in declaration order (for
  `YesNo`, `true` then `false`).
- `confidence::Union{Nothing,Float64}`: the model's confidence, when reported.
- `score::Union{Nothing,Float64}`: for [`Score`](@ref), the expected 0-based ordinal.
- `model`, `usage`: from the response, when reported.
- `raw`: the parsed JSON answer for this question.
"""
struct Answer{T}
    value::T
    probabilities::Union{Nothing,OrderedDict{T,Float64}}
    confidence::Union{Nothing,Float64}
    score::Union{Nothing,Float64}
    model::Union{Nothing,String}
    usage::Any
    raw::Any
end

function Base.show(io::IO, a::Answer{T}) where {T}
    print(io, "Answer{", T, "}(", repr(a.value))
    a.score === nothing || print(io, ", score = ", round(a.score; digits = 3))
    a.confidence === nothing ||
        print(io, ", confidence = ", round(a.confidence; digits = 3))
    return print(io, ")")
end

"""
    ProbabilitiesUnavailable(question)

The model returned only its chosen value, without probabilities. Pass
`allow_point_mass = true` to get a one-hot distribution instead, or use [`decide`](@ref).
"""
struct ProbabilitiesUnavailable <: Exception
    question::Symbol
end

function Base.showerror(io::IO, e::ProbabilitiesUnavailable)
    return print(io, "ProbabilitiesUnavailable: the model returned no probabilities for ",
        "question :", e.question, "; pass allow_point_mass = true for a one-hot map")
end

# Building answers from decoded wire values ------------------------------------------------

_float(x::Real) = Float64(x)
_float(::Nothing) = nothing
_float(x) = throw(DecoderError("expected a number, got $(repr(x))"))

_string(::Nothing) = nothing
_string(x) = string(x)

function _position(opts::Options, name)
    i = get(opts.index, string(name), nothing)
    i === nothing && throw(
        DecoderError(
            "unknown option \"$name\" in response; expected " *
            "one of $(join(opts.names, ", "))",
        ),
    )
    return i
end

# A distribution over `values` from (position => probability) pairs: missing entries are
# 0.0, and the result is renormalized when it does not sum to 1.
function _distribution(values::Vector{T}, entries) where {T}
    ps = zeros(length(values))
    for (i, p) in entries
        1 <= i <= length(ps) || throw(DecoderError("probability for unknown position $i"))
        ps[i] = clamp(_float(p), 0.0, 1.0)
    end
    total = sum(ps)
    total > 0 || throw(DecoderError("probabilities sum to zero"))
    if abs(total - 1) > 1e-6
        abs(total - 1) > 1e-3 &&
            @warn "Decision probabilities sum to $total; renormalizing" maxlog = 10
        ps ./= total
    end
    return OrderedDict{T,Float64}(zip(values, ps))
end

function _yesno_answer(p, model, usage, raw)
    p = clamp(_float(p), 0.0, 1.0)
    return Answer{Bool}(p >= 0.5, OrderedDict(true => p, false => 1 - p), nothing,
        nothing, model, usage, raw)
end

# `probs`: nothing, or (option name => probability) pairs.
function _choice_answer(q::Choice{T}, choice, probs, confidence, model, usage,
    raw) where {T}
    opts = q.options
    dist =
        probs === nothing ? nothing :
        _distribution(opts.values, [_position(opts, k) => p for (k, p) in probs])
    value = if dist !== nothing
        argmax(dist)
    elseif choice !== nothing
        opts.values[_position(opts, choice)]
    else
        throw(DecoderError("choice answer has neither choice nor probabilities"))
    end
    return Answer{T}(value, dist, _float(confidence), nothing, model, usage, raw)
end

# `probs`: nothing, or (0-based ordinal => probability) pairs.
function _score_answer(q::Score{T}, score, probs, confidence, model, usage,
    raw) where {T}
    levels = q.levels.values
    dist =
        probs === nothing ? nothing :
        _distribution(levels, [i + 1 => p for (i, p) in probs])
    if dist !== nothing
        value = argmax(dist)
        score = sum((i - 1) * p for (i, p) in enumerate(values(dist)))
    elseif score !== nothing
        score = _float(score)
        value = levels[clamp(round(Int, score), 0, length(levels)-1)+1]
    else
        throw(DecoderError("score answer has neither score nor probabilities"))
    end
    return Answer{T}(value, dist, _float(confidence), score, model, usage, raw)
end

function _point_mass(a::Answer{T}, q::AbstractQuestion{T}) where {T}
    dist = OrderedDict{T,Float64}(v => Float64(v == a.value) for v in _values(q))
    return Answer{T}(a.value, dist, a.confidence, a.score, a.model, a.usage, a.raw)
end

# Front ends --------------------------------------------------------------------------------

const QuestionSet = NamedTuple{<:Any,<:Tuple{Vararg{AbstractQuestion}}}

function _chunks(names::Tuple, n)
    return [names[i:min(i+n-1, end)] for i in 1:n:length(names)]
end

# `map(f, xs)` with each call in its own task, rethrowing the first failure unwrapped.
# Tasks stay on this thread (`@async`) so user-supplied transports need not be
# thread-safe; the work is I/O-bound.
function _concurrently(f, xs)
    length(xs) == 1 && return [f(only(xs))]
    tasks = [@async f(x) for x in xs]
    for t in tasks
        try
            wait(t)
        catch e
            e isa TaskFailedException || rethrow()
            throw(e.task.exception)
        end
    end
    return map(fetch, tasks)
end

# Send all questions, splitting by `max_questions`, and return a NamedTuple of Answers
# whose probabilities may be `nothing`.
function _ask(backend::AbstractBackend, qs::QuestionSet, state)
    isempty(qs) && throw(ArgumentError("no questions given"))
    check_state(backend, state)
    c = codec(backend)
    extras = request_extras(backend)
    chunks = _chunks(keys(qs), max_questions(backend))
    results = _concurrently(chunks) do names
        sub = NamedTuple{names}(qs)
        body = encode(c, sub, state)
        body["model"] = backend.model
        merge!(body, extras)
        return decode(c, sub, post(backend, body))
    end
    return NamedTuple{keys(qs)}(Tuple(merge(results...)[k] for k in keys(qs)))
end

function _finish(name, a::Answer, q, allow_point_mass)
    a.probabilities === nothing || return a
    allow_point_mass || throw(ProbabilitiesUnavailable(name))
    return _point_mass(a, q)
end

_decide(a::Answer, ::AbstractQuestion, _) = a.value
_decide(a::Answer{Bool}, ::YesNo, threshold) = a.probabilities[true] >= threshold

"""
    ask(backend, question, state; allow_point_mass=false) -> Answer{T}
    ask(backend, (name = question, ...), state; allow_point_mass=false) -> NamedTuple

Ask a decision model `question` about `state` and return the full [`Answer`](@ref). A
`NamedTuple` of questions is sent in one request (split as the backend requires) and
answered with a `NamedTuple` of `Answer`s under the same names.

`state` is a string, any JSON-serializable value, or a `Vector` that may contain
[`Image`](@ref)s when the backend supports them.

Throws [`Decider.ProbabilitiesUnavailable`](@ref) if the model returned only its chosen
value, unless `allow_point_mass = true`, which substitutes a one-hot distribution.
"""
function ask(backend::AbstractBackend, qs::QuestionSet, state; allow_point_mass = false)
    answers = _ask(backend, qs, state)
    names = keys(qs)
    return NamedTuple{names}(
        Tuple(
            _finish(k, answers[k], qs[k], allow_point_mass) for k in names),
    )
end

function ask(backend::AbstractBackend, q::AbstractQuestion, state; allow_point_mass = false)
    return ask(backend, (q = q,), state; allow_point_mass).q
end

"""
    probabilities(backend, question, state; allow_point_mass=false) -> OrderedDict{T,Float64}
    probabilities(backend, (name = question, ...), state; ...) -> NamedTuple

The answer distribution, keyed by your own option type in declaration order. For
[`YesNo`](@ref) the keys are `true` and `false`. See [`ask`](@ref).
"""
function probabilities(backend::AbstractBackend, qs::QuestionSet, state;
    allow_point_mass = false)
    return map(a -> a.probabilities, ask(backend, qs, state; allow_point_mass))
end

function probabilities(backend::AbstractBackend, q::AbstractQuestion, state;
    allow_point_mass = false)
    return ask(backend, q, state; allow_point_mass).probabilities
end

"""
    decide(backend, question, state; threshold=0.5) -> T
    decide(backend, (name = question, ...), state; threshold=0.5) -> NamedTuple

The most likely answer. For [`YesNo`](@ref) this is `p ≥ threshold`. Works even when the
model returns only its chosen value, without probabilities. See [`ask`](@ref).
"""
function decide(backend::AbstractBackend, qs::QuestionSet, state; threshold = 0.5)
    return map((a, q) -> _decide(a, q, threshold), _ask(backend, qs, state), qs)
end

function decide(backend::AbstractBackend, q::AbstractQuestion, state; threshold = 0.5)
    return decide(backend, (q = q,), state; threshold).q
end
