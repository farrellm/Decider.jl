using Aqua
using Decider
using Test

@testset "Decider.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(Decider)
    end
end
