# OpenAI Decisions wire format.
#
# Request:  {"input": ..., "questions": [{"name": id, ...}, ...], "model": ...}
# Response: {"answers": [{"name": id, ...}, ...], "model": ..., "usage": {...}, ...}

function _openai_question(name, q::YesNo)
    if q.yes !== nothing || q.no !== nothing
        msg = "The OpenAI Decisions API has no yes/no criteria; `yes`/`no` are not sent"
        @warn msg maxlog = 1
    end
    return OrderedDict{String,Any}(
        "name" => name, "type" => "predicate", "instructions" => q.instructions)
end

function _labelled(key, names, descriptions)
    return map(names, descriptions) do n, d
        item = OrderedDict{String,Any}(key => n)
        d === nothing || (item["description"] = d)
        return item
    end
end

function _openai_question(name, q::Choice)
    return OrderedDict{String,Any}(
        "name" => name,
        "type" => "choice",
        "instructions" => q.instructions,
        "choices" => _labelled("value", q.options.names, q.options.descriptions),
    )
end

function _openai_question(name, q::Score)
    return OrderedDict{String,Any}(
        "name" => name,
        "type" => "score",
        "instructions" => q.instructions,
        "levels" => _labelled("label", q.levels.names, q.levels.descriptions),
    )
end

_openai_part(s::AbstractString) = Dict("type" => "input_text", "text" => s)
_openai_part(img::Image) = Dict("type" => "input_image", "image_url" => data_url(img))
_openai_part(x) = _openai_part(JSON3.write(x))

_openai_input(state::AbstractString) = state
function _openai_input(state)
    _contains_image(state) || return JSON3.write(state)
    return [Dict("role" => "user", "content" => map(_openai_part, state))]
end

function encode(::OpenAICodec, qs::NamedTuple, state)
    questions = [_openai_question(string(k), q) for (k, q) in pairs(qs)]
    return Dict{String,Any}("input" => _openai_input(state), "questions" => questions)
end

function _openai_answer(::YesNo, a, model, usage)
    p = get(a, :probability, nothing)
    p === nothing && throw(DecoderError("predicate answer has no \"probability\""))
    return _yesno_answer(p, model, usage, a)
end

_entries(::Nothing) = nothing
_entries(x::JSON3.Array) = x
_entries(x) = throw(DecoderError("expected a probabilities array, got $(repr(x))"))

function _openai_answer(q::Choice, a, model, usage)
    probs = _entries(get(a, :probabilities, nothing))
    probs === nothing || (probs = [_string(e.value) => e.probability for e in probs])
    return _choice_answer(q, _string(get(a, :choice, nothing)), probs,
        get(a, :confidence, nothing), model, usage, a)
end

# Score probabilities carry both a 0-based `value` index and a `label`; the label is
# authoritative when present.
function _score_ordinal(q::Score, e)
    label = get(e, :label, nothing)
    label === nothing || return _position(q.levels, label) - 1
    value = get(e, :value, nothing)
    value isa Integer && return Int(value)
    throw(DecoderError("score probability has neither label nor integer value"))
end

function _openai_answer(q::Score, a, model, usage)
    probs = _entries(get(a, :probabilities, nothing))
    probs === nothing || (probs = [_score_ordinal(q, e) => e.probability for e in probs])
    return _score_answer(q, get(a, :score, nothing), probs, get(a, :confidence, nothing),
        model, usage, a)
end

function decode(::OpenAICodec, qs::NamedTuple, json)
    answers = get(json, :answers, nothing)
    answers isa JSON3.Array || throw(DecoderError("response has no \"answers\" array"))
    by_name = Dict(String(a.name) => a for a in answers if haskey(a, :name))
    model = _string(get(json, :model, nothing))
    usage = get(json, :usage, nothing)
    return map(keys(qs), values(qs)) do k, q
        a = get(by_name, string(k), nothing)
        a === nothing && throw(DecoderError("response has no answer for question $k"))
        get(a, :type, nothing) == "refusal" &&
            throw(DecoderError("the model refused to answer question $k"))
        return _openai_answer(q, a, model, usage)
    end |> NamedTuple{keys(qs)}
end
