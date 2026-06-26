# User Interface

```@meta
CurrentModule = ITensorBase
```

This page covers the stable, user-facing API. For the lower-level named-array model, tensor
construction internals and accessors, and experimental features, see the
[Developer Interface](@ref). For a complete alphabetical listing of every documented name,
see the [Reference](@ref).

## Indices and tensors

An [`ITensor`](@ref) labels its dimensions by name, and an [`Index`](@ref) is a named
dimension. Get a tensor's indices with [`inds`](@ref), make distinct copies of an index with
[`prime`](@ref) and [`noprime`](@ref), and mint a fresh unique name with [`uniquename`](@ref).

```@docs; canonical=false
Index
ITensor
inds
prime
noprime
uniquename
```

## Constructors

Build a tensor by calling a `Base` array constructor on indices instead of sizes. `randn`,
`rand`, `zeros`, `ones`, and `fill` all accept indices and return an `ITensor` carrying them.
These are `Base` functions extended to accept indices, so they are shown here by example
rather than in the [Reference](@ref).

```@example userinterface
using ITensorBase: Index
i, j = Index(2), Index(3)
randn(i, j)
```

```@example userinterface
zeros(i, j)
```

## Algebra

ITensors support the standard arithmetic. `*` contracts over shared indices, leaving the
unshared ones.

```@example userinterface
k = Index(2)
a, b = randn(i, j), randn(j, k)
a * b
```

`+` and `-` add and subtract tensors. They are matched up by index, so the operands need not
list their indices in the same order.

```@example userinterface
c = randn(j, i)
a + c
```

Multiplying by a scalar scales the tensor.

```@example userinterface
2 * a
```

## Broadcasting

Linear broadcasting works elementwise and preserves names: adding and subtracting tensors,
and scaling or dividing by a scalar.

```@example userinterface
a .+ c
```

```@example userinterface
2 .* a
```

Non-linear broadcasting (functions of one or more tensors, such as `sin.(a)` or `a .^ 2`) is
experimental and incompletely supported, and is subject to change.
