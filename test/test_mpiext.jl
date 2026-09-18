using ITensorBase: NamedArray, nameddims, unnamed
using MPI: MPI
using Test: @test, @testset

@testset "MPIExt (eltype=$elt)" for elt in (Float64, ComplexF64)
    @testset "Buffer wraps the unnamed parent" begin
        nt = nameddims(randn(elt, (2, 3)), ("i", "j"))
        na = NamedArray(randn(elt, 4), "x")
        for a in (nt, na)
            buffer = MPI.Buffer(a)
            @test buffer.data ≡ unnamed(a)
            @test buffer.count == length(unnamed(a))
            @test buffer.datatype == MPI.Datatype(elt)
        end
    end
    @testset "Sendrecv! round trip" begin
        MPI.Initialized() || MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)

        send = nameddims(randn(elt, (2, 3)), ("i", "j"))
        recv = nameddims(zeros(elt, (2, 3)), ("i", "j"))
        MPI.Sendrecv!(send, recv, comm; dest = rank, source = rank)
        @test unnamed(recv) == unnamed(send)

        send = NamedArray(randn(elt, 4), "x")
        recv = NamedArray(zeros(elt, 4), "x")
        MPI.Sendrecv!(send, recv, comm; dest = rank, source = rank)
        @test unnamed(recv) == unnamed(send)
    end
end
