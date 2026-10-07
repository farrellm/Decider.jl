"""
    Decider

Julia front end for decision models: ask a typed question about some state and get back
probabilities over your own option type (an `@enum` or `Symbol`s), or the most likely
value. Backends: [`TypeSafe`](@ref), [`OpenRouter`](@ref), and [`OpenAIDecisions`](@ref).
"""
module Decider

using Base64: base64encode
using HTTP: HTTP
using JSON3: JSON3
using OrderedCollections: OrderedDict

export ask, decide, probabilities
export Choice, YesNo, Score, Image, Answer
export TypeSafe, OpenRouter, OpenAIDecisions
export DeciderAPIError

include("questions.jl")
include("state.jl")
include("backends.jl")
include("http.jl")
include("answers.jl")
include("codecs/systemone.jl")
include("codecs/openai.jl")

end
