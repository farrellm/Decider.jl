# TypeSafe System One wire format, also spoken by OpenRouter.
#
# Request:  {"state": ..., "questions": {id: question, ...}, "model": ...}
# Response: {"answers": {id: answer, ...}, "model": ..., "usage": {...}, ...}

function _systemone_question(q::YesNo)
    d = OrderedDict{String,Any}("type" => "noul", "instructions" => q.instructions)
    if q.yes !== nothing || q.no !== nothing
        d["criteria"] = OrderedDict("true" => q.yes, "false" => q.no)
    end
    return d
end

function _systemone_question(q::Choice)
    opts = q.options
    return OrderedDict{String,Any}(
        "type" => "choice",
        "instructions" => q.instructions,
        "criteria" => OrderedDict(zip(opts.names, opts.descriptions)),
    )
end

function _systemone_question(q::Score)
    levels = q.levels
    return OrderedDict{String,Any}(
        "type" => "score",
        "instructions" => q.instructions,
        "criteria" =>
            [something(d, n) for (n, d) in zip(levels.names, levels.descriptions)],
    )
end

function encode(::SystemOneCodec, qs::NamedTuple, state)
    questions = OrderedDict{String,Any}(
        string(k) => _systemone_question(q) for (k, q) in pairs(qs))
    return Dict{String,Any}("state" => state, "questions" => questions)
end

_pairs_or_nothing(::Nothing) = nothing
_pairs_or_nothing(x::JSON3.Object) = [String(k) => v for (k, v) in pairs(x)]
_pairs_or_nothing(x) =
    throw(DecoderError("expected a probabilities object, got $(repr(x))"))

function _systemone_answer(q::YesNo, a, model, usage)
    p = get(a, :noul, nothing)
    p === nothing && throw(DecoderError("noul answer has no \"noul\" probability"))
    return _yesno_answer(p, model, usage, a)
end

function _systemone_answer(q::Choice, a, model, usage)
    return _choice_answer(q, _string(get(a, :choice, nothing)),
        _pairs_or_nothing(get(a, :probabilities, nothing)), get(a, :confidence, nothing),
        model, usage, a)
end

function _systemone_answer(q::Score, a, model, usage)
    probs = _pairs_or_nothing(get(a, :probabilities, nothing))
    if probs !== nothing
        probs = map(probs) do (k, p)
            i = tryparse(Int, k)
            i === nothing &&
                throw(DecoderError("score probability key \"$k\" is not an index"))
            return i => p
        end
    end
    return _score_answer(q, get(a, :score, nothing), probs, get(a, :confidence, nothing),
        model, usage, a)
end

function decode(::SystemOneCodec, qs::NamedTuple, json)
    answers = get(json, :answers, nothing)
    answers isa JSON3.Object || throw(DecoderError("response has no \"answers\" object"))
    model = _string(get(json, :model, nothing))
    usage = get(json, :usage, nothing)
    return map(keys(qs), values(qs)) do k, q
        a = get(answers, k, nothing)
        a isa JSON3.Object ||
            throw(DecoderError("response has no answer for question $k"))
        get(a, :type, nothing) == "refusal" &&
            throw(DecoderError("the model refused to answer question $k"))
        return _systemone_answer(q, a, model, usage)
    end |> NamedTuple{keys(qs)}
end
