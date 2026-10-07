# Live API tests: one cheap request per backend whose key is set. Run with
# DECIDER_LIVE_TESTS=1.

@testset "live" begin
    question = (dept = Choice(Dept, "Which team should handle this support ticket?"),
        refund = YesNo("Is the customer asking for money back?"))

    function check(backend)
        a = ask(backend, question, TICKET; allow_point_mass = true)
        @test a.dept.value == billing
        @test sum(values(a.dept.probabilities)) ≈ 1
        @test 0 <= a.refund.probabilities[true] <= 1
    end

    backends = Pair{String,Any}[]
    haskey(ENV, "TYPESAFE_API_KEY") && push!(backends, "TypeSafe" => TypeSafe())
    if haskey(ENV, "OPENROUTER_API_KEY")
        push!(backends, "OpenRouter jev" => OpenRouter(; model = "typesafe/jev-1.13"))
        push!(
            backends,
            "OpenRouter clef-flash" =>
                OpenRouter(; model = "cloudflare/clef-flash"),
        )
    end
    haskey(ENV, "OPENAI_API_KEY") && push!(backends, "OpenAI" => OpenAIDecisions())

    for (name, backend) in backends
        @testset "$name" begin
            check(backend)
        end
    end
end
