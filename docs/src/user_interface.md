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

To name an existing array, index it by the indices: `array[indices...]` returns an `ITensor`
wrapping `array` with one name per dimension. This is the recommended way to turn an unnamed
array into a named one.

```@example userinterface
randn(2, 3)[i, j]
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

Broadcasting works elementwise and matches operands by name, so they need not share index
order. Its advantage over the plain arithmetic above is fusion: a dotted expression is
evaluated in a single pass over the elements, without allocating the intermediate tensors the
undotted form does.

For example, the plain expression below allocates `2 * a` and `3 * c` before adding them:

```@example userinterface
2 * a + 3 * c
```

Adding dots fuses the whole expression into one pass, giving the same result with no
intermediates:

```@example userinterface
2 .* a .+ 3 .* c
```

Non-linear broadcasting (functions of one or more tensors, such as `sin.(a)` or `a .^ 2`) is
experimental and incompletely supported, and is subject to change.

## Factorizations

Matrix factorizations from
[MatrixAlgebraKit](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl) work on an `ITensor`
by naming which indices form the codomain (rows) of the matrix the tensor is interpreted as;
the remaining indices form the domain (columns). The factors share a freshly named index over
the bond between them, so they contract back together with `*`.

A QR decomposition, splitting index `i` off from the rest:

```@example userinterface
using MatrixAlgebraKit: qr_compact
q, r = qr_compact(a, (i,))
q * r ≈ a
```

A singular value decomposition returns three factors:

```@example userinterface
using MatrixAlgebraKit: svd_compact
u, s, v = svd_compact(a, (i,))
u * s * v ≈ a
```

These are MatrixAlgebraKit factorizations wrapped to act on an `ITensor` by name.
MatrixAlgebraKit provides many more, including eigendecompositions, polar decompositions, and
truncated and full variants, along with keyword options controlling them; see the
[MatrixAlgebraKit documentation](https://quantumkithub.github.io/MatrixAlgebraKit.jl/stable/)
for the full list and the keyword syntax. The aim is to wrap all of them, but coverage is
still being filled in, so please open an issue if one you need is not available yet.
