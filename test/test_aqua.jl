using Aqua: Aqua
using ITensorBase: ITensorBase
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(ITensorBase; ambiguities = false, persistent_tasks = false)
end
