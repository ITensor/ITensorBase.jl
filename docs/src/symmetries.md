# Symmetric tensors

```@meta
CurrentModule = ITensorBase
```

ITensorBase supports tensors that are symmetric under group actions by wrapping ITensors around
[GradedArrays.jl](https://github.com/ITensor/GradedArrays.jl). To get started, build
[`Index`](@ref) objects out of `sector => multiplicity` pairs and pass them to the standard Julia
array constructors (`randn`, `zeros`, and so on):

```@example symmetries
using ITensorBase: Index, inds
using GradedArrays: U1, dual, isdual

i = Index([U1(0) => 1, U1(1) => 2])
j = Index([U1(0) => 2, U1(1) => 1])
k = Index([U1(0) => 1, U1(1) => 1])

a = randn(i, dual(j))
```

A `GradedArray` only stores the symmetry-allowed blocks.

These tensors support contraction, multiplication by a scalar, and addition.

```@example symmetries
b = randn(j, dual(k))
a * b
```

```@example symmetries
2 * a
```

```@example symmetries
c = randn(i, dual(j))
a + c
```

## Duality

`dual` flips the duality of an index, and `isdual` returns whether an index is dual. Indices
can only contract with ones that have opposite duality, for example the `j` Index of `b`
contracts with the `dual(j)` Index of `a`.

```@example symmetries
isdual.(inds(a))
```

Note that indices of a `GradedArray` are partitioned into a codomain and a domain, and the
`GradedArray` stores the block diagonal matrix corresponding to the bipartitioning of the
indices. When printing, by convention domain indices are implicitly dual (the format and
conventions are compatible with those from
[TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl)). For more information see the
documentation on [graded arrays](@extref GradedArrays :doc:`user_interface/graded_arrays`).

## Available symmetries

Some standard symmetries are available such as `Z2`, `fU1` (fermionic `U(1)`), and `SU2`. See
[symmetry sectors](@extref GradedArrays Symmetry-sectors) for the complete list and more details.

You can use named sectors to conserve a product of symmetries.

```@example symmetries
Index([(; charge = U1(0), spin = U1(1)) => 1, (; charge = U1(1), spin = U1(0)) => 2])
```
