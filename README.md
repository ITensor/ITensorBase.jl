# ITensorBase.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/ITensorBase.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/ITensorBase.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorBase.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorBase.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorBase.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Installation instructions

The package can be added as usual through the package manager:

```julia
julia> Pkg.add("ITensorBase")
```

## Examples

````julia
using ITensorBase: ITensorBase, ITensor, Index
````

TODO: This should be `TensorAlgebra.qr`.

````julia
using LinearAlgebra: qr
using NamedDimsArrays: NamedDimsArray, aligndims, dimnames, name, unname
using Test: @test
i = Index(2)
j = Index(2)
k = Index(2)
a = randn(i, j)
@test a[j[2], i[1]] == a[1, 2]
@test a[j => 2, i => 1] == a[1, 2]
a′ = randn(j, i)
b = randn(j, k)
c = a * b
@test unname(c, (i, k)) ≈ unname(a, (i, j)) * unname(b, (j, k))
d = a + a′
@test unname(d, (i, j)) ≈ unname(a, (i, j)) + unname(a′, (i, j))
@test a ≈ aligndims(a, (j, i))
q, r = qr(a, (i,))
@test q * r ≈ a
````

Automatic allocation

````julia
a = ITensor(i, j)
````

Broken, need to fix:
a[j[1], i[2]] = 1 + 2im

````julia
a[2, 1] = 1 + 2im
eltype(a) == Complex{Int}
@test a[i[2], j[1]] == 1 + 2im
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

