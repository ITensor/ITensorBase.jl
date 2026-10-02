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

Only the symmetry-allowed blocks are stored.

Tensors over graded indices contract, scale and add like any others.

```@example symmetries
b = randn(j, dual(k))
a * b
```

```@example symmetries
2 * a
```

```@example symmetries
a + randn(i, dual(j))
```

## Duality

`dual` is a GradedArrays function that flips the arrow an index carries, and `isdual` reports
which way it points. A contraction pairs an index with its dual, which is why `b` above is built
over `j` where `a` carries `dual(j)`.

```@example symmetries
isdual.(inds(a))
```

A graded array partitions its indices into a codomain and a domain, the output and input legs,
and stores the block diagonal matrix that bipartitioning gives. Domain indices are implicitly
dual, which is why a domain line in the display of `a * b` above carries no `dual(...)` wrapper
even though `isdual` reports that index as dual. See
[codomain and domain](@extref GradedArrays Codomain-and-domain) for the rest, including how the
conventions line up with TensorKit's.

## Available symmetries

Some standard symmetries are available such as `Z2`, `fU1` (fermionic `U(1)`) and `SU2`. See
[symmetry sectors](@extref GradedArrays Symmetry-sectors) for the complete list and more details.

Conserving more than one quantity at once means a product of symmetries, written as a
`NamedTuple` of sectors naming each factor.

```@example symmetries
Index([(; charge = U1(0), spin = U1(1)) => 1, (; charge = U1(1), spin = U1(0)) => 2])
```
