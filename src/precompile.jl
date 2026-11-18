using LinearAlgebra: norm
using MatrixAlgebraKit: svd_trunc
using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        i, j, k = Index(2; tags = "i"), Index(2; tags = "j"), Index(2; tags = "k")
        for T in (Float64, ComplexF64)
            a = randn(T, i, j, k)
            b = randn(T, (k,), (prime(i),))
            conj(a) * b
            d = rename(a * b, prime(i) => k)
            noprime(prime(d))
            d = d / norm(d)
            u, s, v, _ = svd_trunc(d, (i, j), (k,); trunc = (; maxrank = 2))
            u * s * v
            commonind(a, b)
        end
    end
end
