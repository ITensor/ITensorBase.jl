# Symmetric tensors

```@meta
CurrentModule = ITensorBase
```

An [`Index`](@ref) can carry a graded space, which makes a tensor over it block sparse and makes
contraction conserve the symmetry. The spaces and the sectors that grade them come from
GradedArrays, whose [symmetry sectors](@extref GradedArrays Symmetry-sectors) page lists the
symmetries available.

Pass `sector => multiplicity` pairs where an ungraded index takes a length.

```@example symmetries
using ITensorBase: ITensor, Index, inds
using GradedArrays: U1, dual, isdual

i = Index([U1(0) => 1, U1(1) => 2])
```

## Duality

A contraction pairs an index with a dual one, so an index carries an arrow. `dual` turns it
around, and `conj` is an alternative spelling of the same thing.

```@example symmetries
dual(i)
```

```@example symmetries
conj(i) == dual(i)
```

A tensor over graded indices stores only the symmetry-allowed blocks.

```@example symmetries
j = Index([U1(0) => 2, U1(1) => 1])
a = randn(i, dual(j))
```

`isdual` reports an index's arrow, which is how to read a tensor's duality off its legs.

```@example symmetries
isdual.(inds(a))
```
