"""
    AbstractQuestion{T}

A question for a decision model. `T` is the type of the answer's values: the option type
for [`Choice`](@ref), the level type for [`Score`](@ref), and `Bool` for [`YesNo`](@ref).
"""
abstract type AbstractQuestion{T} end

"""
    describe(x)::Union{Nothing,String}

Description sent to the model for the `Enum` option or `Score` level `x`, used when no
`descriptions` were passed to [`Choice`](@ref). Defaults to `nothing`; overload it for your
own enum:

```julia
Decider.describe(x::Dept) = x == billing ? "Charges, refunds, invoices" : nothing
```
"""
describe(::Any) = nothing

# The options of a Choice or the levels of a Score, in declaration order. `names` are the
# wire strings (`string(x)`), and `index` inverts them to positions in `values`.
struct Options{T}
    values::Vector{T}
    names::Vector{String}
    descriptions::Vector{Union{Nothing,String}}
    index::Dict{String,Int}
end

function Options(values::Vector{T}, descriptions::Vector{Union{Nothing,String}},
    what::AbstractString, range::UnitRange{Int}) where {T}
    n = length(values)
    n in range ||
        throw(ArgumentError("$what needs $(first(range))–$(last(range)) entries, got $n"))
    names = String[string(v) for v in values]
    index = Dict{String,Int}()
    for (i, name) in pairs(names)
        isempty(name) && throw(ArgumentError("$what names must be non-empty"))
        haskey(index, name) && throw(ArgumentError("duplicate $what name \"$name\""))
        index[name] = i
    end
    return Options{T}(values, names, descriptions, index)
end

_description(::Nothing) = nothing
_description(s::AbstractString) = String(s)

function _lookup_descriptions(values::Vector{T}, descriptions) where {T}
    descriptions === nothing &&
        return Union{Nothing,String}[_description(describe(v)) for v in values]
    table = Dict{T,Union{Nothing,String}}()
    for (k, v) in descriptions
        k isa T && k in values ||
            throw(ArgumentError("description given for unknown option $(repr(k))"))
        table[k] = _description(v)
    end
    return Union{Nothing,String}[get(table, v, nothing) for v in values]
end

function _options(E::Type{<:Enum}, descriptions, what, range)
    values = collect(instances(E))
    return Options(values, _lookup_descriptions(values, descriptions), what, range)
end

_option_entry(x::Symbol) = x => nothing
_option_entry(p::Pair{Symbol}) = first(p) => _description(last(p))
_option_entry(x) = throw(
    ArgumentError(
        "options must be Symbols or Symbol => description pairs, got $(repr(x))"),
)

# A Vector of Symbols and/or `Symbol => description` pairs.
function _options(xs::AbstractVector, descriptions, what, range)
    entries = map(_option_entry, xs)
    values = Symbol[first(e) for e in entries]
    if any(e -> last(e) !== nothing, entries)
        descriptions === nothing || throw(
            ArgumentError(
                "pass descriptions either as pairs or as a keyword, not both"),
        )
        descs = Union{Nothing,String}[last(e) for e in entries]
    else
        descs = _lookup_descriptions(values, descriptions)
    end
    return Options(values, descs, what, range)
end

_options(xs, _, what, _) = throw(
    ArgumentError(
        "$what must be an Enum type or a Vector of Symbols and/or Symbol => description " *
        "pairs; got $(typeof(xs))"),
)

"""
    Choice(E::Type{<:Enum}, instructions; descriptions=nothing)
    Choice([:a, :b, :c], instructions; descriptions=nothing)
    Choice([:a => "description", :b => "description", :c], instructions)

Ask the model to pick one of several options. The answer type `T` is `E` for an enum and
`Symbol` otherwise; options are kept in declaration order.

`instructions` is a `String` or any JSON-serializable value. Option descriptions come from
the pairs, the `descriptions` keyword (a `Dict` or pairs keyed by option), or
[`Decider.describe`](@ref). Options are sent as `string(x)`; there must be 2–255 of them,
with distinct non-empty names.
"""
struct Choice{T} <: AbstractQuestion{T}
    instructions::Any
    options::Options{T}
end

function Choice(options, instructions; descriptions = nothing)
    return Choice(instructions, _options(options, descriptions, "Choice options", 2:255))
end

"""
    Score(levels, instructions; descriptions=nothing)

Ask the model to rate on an ordered scale. `levels` is an `Enum` type (declaration order is
the ordinal) or a `Vector` of `Symbol`s and/or `Symbol => description` pairs, with 2–10
levels. Descriptions work as for [`Choice`](@ref); a level without one is described by its
name.

The [`Answer`](@ref) carries `.score`, the expected ordinal (0-based), and [`decide`](@ref)
returns the most likely level. When you want a single level that respects the whole
distribution, `levels[round(Int, a.score) + 1]` is often the better choice.
"""
struct Score{T} <: AbstractQuestion{T}
    instructions::Any
    levels::Options{T}
end

function Score(levels, instructions; descriptions = nothing)
    return Score(instructions, _options(levels, descriptions, "Score levels", 2:10))
end

"""
    YesNo(instructions; yes=nothing, no=nothing)

Ask a yes/no question. `yes` and `no` optionally describe what counts as each answer.
[`probabilities`](@ref) returns `true => p, false => 1 - p`, and [`decide`](@ref) returns
`p ≥ threshold` (default 0.5).
"""
struct YesNo <: AbstractQuestion{Bool}
    instructions::Any
    yes::Union{Nothing,String}
    no::Union{Nothing,String}
end

function YesNo(instructions; yes = nothing, no = nothing)
    return YesNo(instructions, _description(yes), _description(no))
end

_values(q::Choice) = q.options.values
_values(q::Score) = q.levels.values
_values(::YesNo) = [true, false]
