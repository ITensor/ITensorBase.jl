# ITensorBase.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/ITensorBase.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/ITensorBase.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorBase.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorBase.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorBase.jl)
[![Code Style](https://img.shields.io/badge/code_style-ITensor-purple)](https://github.com/ITensor/ITensorFormatter.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Support

<picture>
  <source media="(prefers-color-scheme: dark)" width="20%" srcset="docs/src/assets/CCQ-dark.png">
  <img alt="Flatiron Center for Computational Quantum Physics logo." width="20%" src="docs/src/assets/CCQ.png">
</picture>


ITensorBase.jl is supported by the Flatiron Institute, a division of the Simons Foundation.

## Installation instructions

This package resides in the `ITensor/ITensorRegistry` local registry.
In order to install, simply add that registry through your package manager.
This step is only required once.
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
or:
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

Then, the package can be added as usual through the package manager:

```julia
julia> Pkg.add("ITensorBase")
```

## Examples

Load the package, along with a factorization from
[MatrixAlgebraKit.jl](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl).

````julia
using ITensorBase: Index
using MatrixAlgebraKit: qr_compact
````

An `Index` labels one dimension of a tensor and carries its length. Each call makes a new,
distinct index, so a tensor identifies its dimensions by index rather than by position.

````julia
i = Index(2)
j = Index(2)
k = Index(2)
````

Make a random `ITensor` with indices `i` and `j`.

````julia
a = randn(i, j)
````

Read off an element by giving each index a value with `i[value]`. The indices can be given
in any order, since elements are looked up by index, not by position.

````julia
a[j[2], i[1]]
````

Contract `a` with another tensor over their shared index `j`. `j` is summed over and the
result keeps the remaining indices `i` and `k`.

````julia
b = randn(j, k)
a * b
````

Add two tensors. They are matched up by index, so `a` and `c` don't need their indices in
the same order.

````julia
c = randn(j, i)
a + c
````

Factorize `a` over index `i` into a `q` with orthonormal columns and an upper-triangular
`r`. The factors share a new index that `qr_compact` introduces.

````julia
q, r = qr_compact(a, (i,))
````

Contracting the factors back together recovers `a`.

````julia
q * r
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

