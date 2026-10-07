using Aqua
using Decider
using Decider: DecoderError, ProbabilitiesUnavailable
using JSON3
using OrderedCollections: OrderedDict
using Test

@enum Dept billing technical sales
@enum Severity low medium high

Decider.describe(x::Severity) = x == high ? "Outage or data loss" : nothing

# A transport that records requests and replays canned responses, in order. Each response
# is a JSON string (status 200), a (status, body) tuple, a (status, headers, body) tuple,
# or an exception to throw.
mutable struct MockTransport
    responses::Vector{Any}
    requests::Vector{Any}
end
mock(responses...) = MockTransport(collect(Any, responses), Any[])

function (t::MockTransport)(method, url, headers, body)
    push!(t.requests, (; method, url, headers, body = JSON3.read(body)))
    r = popfirst!(t.responses)
    r isa Exception && throw(r)
    r isa AbstractString && return 200, Pair{String,String}[], r
    length(r) == 2 && return r[1], Pair{String,String}[], r[2]
    return r
end

json(x) = JSON3.read(JSON3.write(x))

const SLEEPS = Float64[]
Decider.SLEEP[] = s -> push!(SLEEPS, s)

const TICKET = "I was charged twice for my subscription."

@testset "Decider.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(Decider)
    end

    @testset "questions" begin
        q = Choice(Dept, "Which team?")
        @test q isa Decider.AbstractQuestion{Dept}
        @test q.options.values == [billing, technical, sales]
        @test q.options.names == ["billing", "technical", "sales"]
        @test all(isnothing, q.options.descriptions)

        q = Choice(Dept, "Which team?"; descriptions = Dict(billing => "Money"))
        @test q.options.descriptions == ["Money", nothing, nothing]
        q = Choice(Dept, "Which team?"; descriptions = [sales => "Buying"])
        @test q.options.descriptions == [nothing, nothing, "Buying"]

        q = Choice([:a, :b], "?")
        @test q isa Choice{Symbol}
        q = Choice([:a => "first", :b => "second"], "?")
        @test q.options.values == [:a, :b]
        @test q.options.descriptions == ["first", "second"]

        s = Score(Severity, "How bad?")
        @test s isa Decider.AbstractQuestion{Severity}
        @test s.levels.descriptions == [nothing, nothing, "Outage or data loss"]

        y = YesNo("Urgent?"; yes = "Needs action today")
        @test y isa Decider.AbstractQuestion{Bool}
        @test y.yes == "Needs action today" && y.no === nothing

        q = Choice(Dept, (task = "route", context = "support desk"))
        @test q.instructions == (task = "route", context = "support desk")

        @test_throws ArgumentError Choice([:a], "?")
        @test_throws ArgumentError Choice([:a, :a], "?")
        @test_throws ArgumentError Choice([:a, Symbol("")], "?")
        @test_throws ArgumentError Choice([Symbol("o$i") for i in 1:256], "?")
        @test length(Choice([Symbol("o$i") for i in 1:255], "?").options.values) == 255
        @test_throws ArgumentError Score([Symbol("l$i") for i in 1:11], "?")
        @test_throws ArgumentError Score([:only], "?")
        @test_throws ArgumentError Choice(Dept, "?"; descriptions = Dict(low => "x"))
        @test_throws ArgumentError Choice([:a => "x", :b => "y"], "?"; descriptions = [])
        @test_throws ArgumentError Choice(["a", "b"], "?")
        @test_throws ArgumentError Choice(Dept, "?"; descriptions = Dict(:billing => "x"))
        s = Score([:low => "Cosmetic", :mid, :high], "?")
        @test s.levels.descriptions == ["Cosmetic", nothing, nothing]
    end

    @testset "backends" begin
        withenv("TYPESAFE_API_KEY" => nothing, "OPENROUTER_API_KEY" => nothing,
            "OPENAI_API_KEY" => nothing) do
            b = TypeSafe(; transport = mock())
            @test_throws ArgumentError decide(b, YesNo("?"), "x")
            @test isempty(b.transport.requests)
            @test OpenRouter() isa OpenRouter
            @test OpenAIDecisions() isa OpenAIDecisions
        end
        withenv("TYPESAFE_API_KEY" => "env-key") do
            @test Decider.headers(TypeSafe())[1] == ("Authorization" => "Bearer env-key")
        end
        b = TypeSafe(; api_key = "secret-key")
        @test Decider.headers(b)[1] == ("Authorization" => "Bearer secret-key")
        @test !occursin("secret", repr(b))
        @test Decider.endpoint_url(b) == "https://api.typesafe.ai/v1/systemone"
        @test Decider.endpoint_url(TypeSafe(; base_url = "http://localhost:8000/")) ==
              "http://localhost:8000/v1/systemone"

        r = OpenRouter(; api_key = "k", provider = (order = ["typesafe"],), user = "u1")
        @test Decider.endpoint_url(r) == "https://openrouter.ai/api/alpha/decisions"
        @test Decider.request_extras(r) ==
              Dict{String,Any}("provider" => (order = ["typesafe"],), "user" => "u1")
        r = OpenRouter(; endpoint = :systemone)
        @test Decider.endpoint_url(r) == "https://openrouter.ai/api/v1/systemone"
        @test_throws ArgumentError OpenRouter(; endpoint = :systemone, session_id = "s")
        @test_throws ArgumentError OpenRouter(; endpoint = :chat)

        o = OpenAIDecisions()
        @test Decider.endpoint_url(o) == "https://api.openai.com/v1/decisions"
        @test Decider.max_questions(o) == 10
        @test Decider.max_questions(TypeSafe()) == typemax(Int)
        @test Decider.supports_images(o)
        @test !Decider.supports_images(TypeSafe()) && !Decider.supports_images(OpenRouter())
    end

    @testset "SystemOne encoding" begin
        qs = (
            dept = Choice(Dept, "Which team should handle this?";
                descriptions = Dict(billing => "Charges and refunds")),
            refund = YesNo("Is the customer asking for a refund?";
                yes = "Asks for money back or a credit",
                no = "Mentions a charge without asking for money back"),
            plain = YesNo("Is this spam?"),
            severity = Score(
                [:low => "Cosmetic", :medium => "Degraded", :high],
                "How bad?",
            ),
        )
        t = mock(
            """{"answers": {
       "dept": {"type": "choice", "choice": "billing",
                "probabilities": {"billing": 0.9, "technical": 0.1, "sales": 0.0},
                "confidence": 0.9},
       "refund": {"type": "noul", "noul": 0.2},
       "plain": {"type": "noul", "noul": 0.01},
       "severity": {"type": "score", "score": 0.5, "legend": {"0": "Cosmetic", "1": "Degraded", "2": "high"},
                    "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0},
                    "confidence": 0.5}}}""",
        )
        decide(TypeSafe(; api_key = "k", transport = t), qs, TICKET)
        req = only(t.requests)
        @test req.method == "POST"
        @test req.url == "https://api.typesafe.ai/v1/systemone"
        @test ("Authorization" => "Bearer k") in req.headers
        @test req.body == json(
            Dict(
                "model" => "jev-latest",
                "state" => TICKET,
                "questions" => Dict(
                    "dept" => Dict(
                        "type" => "choice",
                        "instructions" => "Which team should handle this?",
                        "criteria" => Dict("billing" => "Charges and refunds",
                            "technical" => nothing, "sales" => nothing)),
                    "refund" => Dict(
                        "type" => "noul",
                        "instructions" => "Is the customer asking for a refund?",
                        "criteria" =>
                            Dict("true" => "Asks for money back or a credit",
                                "false" => "Mentions a charge without asking for money back",
                            )),
                    "plain" =>
                        Dict("type" => "noul", "instructions" => "Is this spam?"),
                    "severity" => Dict(
                        "type" => "score",
                        "instructions" => "How bad?",
                        "criteria" => ["Cosmetic", "Degraded", "high"]),
                ),
            ),
        )
        # Choice criteria keep declaration order on the wire.
        @test collect(keys(req.body.questions.dept.criteria)) ==
              [:billing, :technical, :sales]

        # Structured state and instructions, and the single-question id "q".
        t = mock("""{"answers": {"q": {"type": "noul", "noul": 0.5}}}""")
        state = Dict("ticket" => TICKET, "plan" => "pro", "history" => [1, 2])
        decide(TypeSafe(; api_key = "k", transport = t), YesNo((rule = "is pro",)), state)
        req = only(t.requests)
        @test req.body.state == json(state)
        @test req.body.questions.q.instructions == json((rule = "is pro",))

        # OpenRouter extras are merged into the body.
        t = mock("""{"answers": {"q": {"type": "noul", "noul": 0.5}}}""")
        r = OpenRouter(; api_key = "k", model = "cloudflare/clef-flash",
            provider = (order = ["cloudflare"],), session_id = "s1", transport = t)
        decide(r, YesNo("?"), "x")
        req = only(t.requests)
        @test req.url == "https://openrouter.ai/api/alpha/decisions"
        @test req.body.model == "cloudflare/clef-flash"
        @test req.body.provider == json((order = ["cloudflare"],))
        @test req.body.session_id == "s1"
        @test !haskey(req.body, :user)
    end

    @testset "SystemOne decoding" begin
        # TypeSafe response
        typesafe = """{
          "model": "jev-1.13",
          "answers": {
            "dept": {"type": "choice", "choice": "billing",
                     "probabilities": {"billing": 0.88, "technical": 0.12},
                     "confidence": 0.91},
            "urgent": {"type": "noul", "noul": 0.87},
            "severity": {"type": "score", "score": 1.5, "legend": {"0": "low", "1": "medium", "2": "high"},
                         "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                         "confidence": 0.76}
          },
          "usage": {"input_tokens": 42, "output_tokens": 3}
        }"""
        qs =
            (dept = Choice(Dept, "?"), urgent = YesNo("?"), severity = Score(Severity, "?"))
        b = TypeSafe(; api_key = "k", transport = mock(typesafe))
        a = ask(b, qs, TICKET)
        @test keys(a) == (:dept, :urgent, :severity)
        @test a.dept isa Answer{Dept}
        @test a.dept.value == billing
        @test a.dept.probabilities ==
              OrderedDict(billing => 0.88, technical => 0.12, sales => 0.0)
        @test collect(keys(a.dept.probabilities)) == [billing, technical, sales]
        @test a.dept.confidence == 0.91
        @test a.dept.model == "jev-1.13"
        @test a.dept.usage.input_tokens == 42
        @test a.dept.raw.choice == "billing"
        @test a.urgent.value === true
        @test a.urgent.probabilities == OrderedDict(true => 0.87, false => 1 - 0.87)
        @test a.severity.value == high
        @test a.severity.score ≈ 1.5
        @test a.severity.confidence == 0.76
        @test collect(values(a.severity.probabilities)) == [0.1, 0.3, 0.6]

        b = TypeSafe(; api_key = "k", transport = mock(typesafe, typesafe))
        @test decide(b, qs, TICKET) == (dept = billing, urgent = true, severity = high)
        p = probabilities(b, qs, TICKET)
        @test p.urgent isa AbstractDict{Bool,Float64}
        @test p.dept[sales] == 0.0

        # OpenRouter response: extra id/provider fields and usage.cost are tolerated.
        openrouter = """{
          "id": "gen-123", "provider": "TypeSafe", "model": "typesafe/jev-1.13",
          "answers": {"q": {"type": "choice", "choice": "b",
                            "probabilities": {"a": 0.25, "b": 0.75}, "confidence": 0.8}},
          "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.00001}
        }"""
        r = OpenRouter(; api_key = "k", transport = mock(openrouter))
        a = ask(r, Choice([:a, :b], "?"), "x")
        @test a isa Answer{Symbol}
        @test a.value == :b
        @test a.usage.cost == 0.00001
        @test a.model == "typesafe/jev-1.13"
    end

    @testset "OpenAI encoding" begin
        t = mock("""{"answers": [
            {"name": "dept", "type": "choice", "choice": "sales"},
            {"name": "urgent", "type": "predicate", "probability": 0.4},
            {"name": "severity", "type": "score", "score": 0}]}""")
        qs = (dept = Choice([:billing => "Money", :sales], "Which team?"),
            urgent = YesNo("Is this urgent?"),
            severity = Score(Severity, "How bad?"))
        decide(OpenAIDecisions(; api_key = "k", transport = t), qs, TICKET)
        req = only(t.requests)
        @test req.url == "https://api.openai.com/v1/decisions"
        @test req.body == json(
            Dict(
                "model" => "gpt-6-luna",
                "input" => TICKET,
                "questions" => [
                    Dict("name" => "dept", "type" => "choice",
                        "instructions" => "Which team?",
                        "choices" =>
                            [Dict("value" => "billing", "description" => "Money"),
                                Dict("value" => "sales")]),
                    Dict("name" => "urgent", "type" => "predicate",
                        "instructions" => "Is this urgent?"),
                    Dict("name" => "severity", "type" => "score",
                        "instructions" => "How bad?",
                        "levels" =>
                            [Dict("label" => "low"), Dict("label" => "medium"),
                                Dict(
                                    "label" => "high",
                                    "description" => "Outage or data loss",
                                )]),
                ],
            ),
        )

        # Structured state is sent as JSON text; images switch to input messages.
        t = mock(
            """{"answers": [{"name": "q", "type": "predicate", "probability": 0.5}]}""",
            """{"answers": [{"name": "q", "type": "predicate", "probability": 0.5}]}""")
        o = OpenAIDecisions(; api_key = "k", transport = t)
        decide(o, YesNo("?"), Dict("a" => 1))
        @test t.requests[1].body.input == """{"a":1}"""
        img = Image(UInt8[0x89, 0x50], "image/png")
        decide(o, YesNo("Is there a cat?"), ["Photo:", img, Dict("exif" => "none")])
        @test t.requests[2].body.input == json([
            Dict(
                "role" => "user",
                "content" => [
                    Dict("type" => "input_text", "text" => "Photo:"),
                    Dict(
                        "type" => "input_image",
                        "image_url" => "data:image/png;base64,iVA=",
                    ),
                    Dict("type" => "input_text", "text" => """{"exif":"none"}"""),
                ],
            ),
        ])

        # yes/no criteria have no OpenAI equivalent.
        t = mock(
            """{"answers": [{"name": "q", "type": "predicate", "probability": 0.5}]}""")
        o = OpenAIDecisions(; api_key = "k", transport = t)
        @test_logs (:warn, r"no yes/no criteria") decide(o, YesNo("?"; yes = "y"), "x")
        @test !haskey(only(t.requests).body.questions[1], :criteria)
    end

    @testset "OpenAI decoding" begin
        response = """{
          "id": "dec_123", "model": "gpt-6-luna",
          "answers": [
            {"name": "urgent", "type": "predicate", "probability": 0.87},
            {"name": "dept", "type": "choice", "choice": "billing", "confidence": 0.91,
             "probabilities": [{"value": "billing", "probability": 0.91},
                               {"value": "technical", "probability": 0.06},
                               {"value": "sales", "probability": 0.03}]},
            {"name": "severity", "type": "score", "score": 0.4, "confidence": 0.7,
             "probabilities": [{"value": 0, "label": "low", "probability": 0.7},
                               {"value": 1, "label": "medium", "probability": 0.2},
                               {"value": 2, "label": "high", "probability": 0.1}]}
          ],
          "usage": {"input_tokens": 50, "output_tokens": 3}
        }"""
        qs =
            (dept = Choice(Dept, "?"), urgent = YesNo("?"), severity = Score(Severity, "?"))
        o = OpenAIDecisions(; api_key = "k", transport = mock(response))
        a = ask(o, qs, TICKET)
        @test a.dept.value == billing
        @test a.dept.probabilities[sales] == 0.03
        @test a.dept.model == "gpt-6-luna"
        @test a.urgent.probabilities[true] == 0.87
        @test a.severity.value == low
        @test a.severity.score ≈ 0.4
        @test a.severity.usage.output_tokens == 3

        # Score levels map by integer value when labels are absent.
        response = """{"answers": [{"name": "q", "type": "score", "score": 2,
            "probabilities": [{"value": 2, "probability": 1.0}]}]}"""
        o = OpenAIDecisions(; api_key = "k", transport = mock(response))
        @test decide(o, Score(Severity, "?"), "x") == high

        response = """{"answers": [{"name": "q", "type": "refusal"}]}"""
        o = OpenAIDecisions(; api_key = "k", transport = mock(response))
        @test_throws DecoderError decide(o, YesNo("?"), "x")

        response = """{"answers": [{"name": "other", "type": "predicate", "probability": 1}]}"""
        o = OpenAIDecisions(; api_key = "k", transport = mock(response))
        @test_throws DecoderError decide(o, YesNo("?"), "x")
    end

    @testset "OpenAI chunking" begin
        names = Tuple(Symbol("q$i") for i in 1:23)
        qs = NamedTuple{names}(Tuple(YesNo("Question $i?") for i in 1:23))
        reply(range) = JSON3.write(
            Dict(
                "answers" => [
                    Dict("name" => "q$i", "type" => "predicate", "probability" => i / 100)
                    for i in range],
            ),
        )
        # Chunks are sent concurrently, so answer each request with what it asked for.
        requests = Any[]
        lk = ReentrantLock()
        transport = function (method, url, headers, body)
            req = JSON3.read(body)
            lock(() -> push!(requests, req), lk)
            ids = [parse(Int, q.name[2:end]) for q in req.questions]
            return 200, Pair{String,String}[], reply(ids)
        end
        p = probabilities(OpenAIDecisions(; api_key = "k", transport), qs, "x")
        @test keys(p) == names
        @test [p[n][true] for n in names] ≈ (1:23) ./ 100
        @test sort(length.(getproperty.(requests, :questions))) == [3, 10, 10]

        # SystemOne backends send everything at once.
        t = mock(
            JSON3.write(
                Dict(
                    "answers" => Dict(
                        "q$i" => Dict("type" => "noul", "noul" => 0.5) for i in 1:23
                    ),
                ),
            ),
        )
        decide(TypeSafe(; api_key = "k", transport = t), qs, "x")
        @test length(only(t.requests).body.questions) == 23
    end

    @testset "normalization" begin
        reply(probs) = JSON3.write(
            Dict(
                "answers" => Dict(
                    "q" => Dict(
                        "type" => "choice", "choice" => "a",
                        "probabilities" => probs),
                ),
            ),
        )
        backend(probs...) = TypeSafe(; api_key = "k",
            transport = mock((reply(p) for p in probs)...))
        q = Choice([:a, :b, :c], "?")

        p = @test_logs probabilities(backend(Dict("a" => 0.6, "b" => 0.40001)), q, "x")
        @test sum(values(p)) ≈ 1 atol = 1e-12
        @test p[:a] ≈ 0.6 / 1.00001
        @test p[:c] == 0.0

        p = @test_logs (:warn, r"renormalizing") probabilities(
            backend(Dict("a" => 0.3, "b" => 0.3)), q, "x")
        @test p == OrderedDict(:a => 0.5, :b => 0.5, :c => 0.0)

        @test_throws DecoderError probabilities(
            backend(Dict("a" => 0.5, "z" => 0.5)),
            q,
            "x",
        )
        @test_throws DecoderError probabilities(backend(Dict("a" => 0.0)), q, "x")

        # decide is the argmax of the distribution.
        @test decide(backend(Dict("a" => 0.2, "b" => 0.5, "c" => 0.3)), q, "x") == :b

        # Noul probabilities are clamped to [0, 1].
        noul(p) = TypeSafe(;
            api_key = "k",
            transport = mock(
                JSON3.write(
                    Dict("answers" => Dict("q" => Dict("type" => "noul", "noul" => p))),
                ),
            ),
        )
        @test probabilities(noul(1.2), YesNo("?"), "x")[true] == 1.0
        @test probabilities(noul(-0.1), YesNo("?"), "x")[false] == 1.0
        @test decide(noul(0.5), YesNo("?"), "x") === true
        @test decide(noul(0.7), YesNo("?"), "x"; threshold = 0.8) === false
        @test decide(noul(0.7), (q = YesNo("?"),), "x"; threshold = 0.6) == (q = true,)
    end

    @testset "missing probabilities" begin
        reply = """{"answers": {
            "dept": {"type": "choice", "choice": "technical"},
            "sev": {"type": "score", "score": 1.4}}}"""
        backend() = TypeSafe(; api_key = "k", transport = mock(reply))
        qs = (dept = Choice(Dept, "?"), sev = Score(Severity, "?"))
        @test decide(backend(), qs, "x") == (dept = technical, sev = medium)
        err = try
            probabilities(backend(), qs, "x")
        catch e
            e
        end
        @test err isa ProbabilitiesUnavailable && err.question == :dept
        @test occursin("allow_point_mass", sprint(showerror, err))
        @test_throws ProbabilitiesUnavailable ask(backend(), qs, "x")
        p = probabilities(backend(), qs, "x"; allow_point_mass = true)
        @test p.dept == OrderedDict(billing => 0.0, technical => 1.0, sales => 0.0)
        @test p.sev == OrderedDict(low => 0.0, medium => 1.0, high => 0.0)
        a = ask(backend(), qs, "x"; allow_point_mass = true).sev
        @test a.score == 1.4

        neither = """{"answers": {"q": {"type": "choice", "confidence": 0.5}}}"""
        b = TypeSafe(; api_key = "k", transport = mock(neither))
        @test_throws DecoderError decide(b, Choice(Dept, "?"), "x")
        b = TypeSafe(; api_key = "k", transport = mock("""{"answers": {}}"""))
        @test_throws DecoderError decide(b, Choice(Dept, "?"), "x")
        b = TypeSafe(; api_key = "k", transport = mock("""{"ok": true}"""))
        @test_throws DecoderError decide(b, Choice(Dept, "?"), "x")
        b = TypeSafe(; api_key = "k", transport = mock("not json"))
        @test_throws DecoderError decide(b, Choice(Dept, "?"), "x")
    end

    @testset "retries" begin
        ok = """{"answers": {"q": {"type": "noul", "noul": 0.9}}}"""
        retry(responses...; retries = 3) =
            TypeSafe(; api_key = "k", retries, transport = mock(responses...))

        empty!(SLEEPS)
        b = retry((429, ["Retry-After" => "7"], "slow down"), (529, "overloaded"),
            (503, ""), ok)
        @test decide(b, YesNo("?"), "x") === true
        @test length(b.transport.requests) == 4
        @test SLEEPS[1] == 7.0
        @test length(SLEEPS) == 3

        b = retry(Base.IOError("connection reset", 0), ok)
        @test decide(b, YesNo("?"), "x") === true

        for status in (400, 401, 402, 422)
            b = retry((status, """{"error": {"message": "bad request $status"}}"""), ok)
            err = try
                decide(b, YesNo("?"), "x")
            catch e
                e
            end
            @test err isa DeciderAPIError
            @test err.status == status
            @test err.message == "bad request $status"
            @test length(b.transport.requests) == 1
        end

        b = retry((429, "a"), (429, "b"), (429, """{"detail": "rate limited"}""");
            retries = 2)
        err = try
            decide(b, YesNo("?"), "x")
        catch e
            e
        end
        @test err isa DeciderAPIError && err.status == 429
        @test err.message == "rate limited"
        @test occursin("HTTP 429", sprint(showerror, err))

        b = retry(ErrorException("bug in transport"), ok)
        @test_throws ErrorException decide(b, YesNo("?"), "x")
    end

    @testset "images" begin
        img = Image(UInt8[1, 2, 3], MIME("image/jpeg"))
        @test img.mime == "image/jpeg"
        @test Decider.data_url(img) == "data:image/jpeg;base64,AQID"
        mktempdir() do dir
            path = joinpath(dir, "cat.PNG")
            write(path, UInt8[1, 2])
            @test Image(path).mime == "image/png"
            @test_throws ArgumentError Image(joinpath(dir, "cat.bmp"))
        end

        state = ["Is there a cat?", img]
        for b in (TypeSafe(; api_key = "k", transport = mock()),
            OpenRouter(; api_key = "k", transport = mock()))
            @test_throws ArgumentError decide(b, YesNo("?"), state)
            @test isempty(b.transport.requests)
        end
        o = OpenAIDecisions(; api_key = "k", transport = mock())
        @test_throws ArgumentError decide(o, YesNo("?"), Dict("photo" => img))
        @test_throws ArgumentError decide(o, YesNo("?"), img)
        @test_throws ArgumentError decide(o, YesNo("?"), ["x", [img]])
        @test isempty(o.transport.requests)
    end

    if get(ENV, "DECIDER_LIVE_TESTS", "") == "1"
        include("live.jl")
    end
end
