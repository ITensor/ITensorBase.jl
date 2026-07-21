# Developer Interface

```@meta
CurrentModule = ITensorBase
```

This page covers the named-array model that indices and tensors are built on, the interface
for implementing a new tensor type, and experimental features that are not yet part of the
stable user-facing API. For the stable user-facing API, see the [User Interface](@ref).

## Named array types

A concrete tensor type subtypes [`AbstractNamedTensor`](@ref). [`NamedTensor`](@ref)
is the built-in implementation, and [`ITensor`](@ref) is the `NamedTensor` with dimension
names that are [`IndexName`](@ref)s. Its `NamedTensor(array, dimnames)` constructor pairs an array of
any kind with its dimension names directly. User code usually builds one by calling an array constructor on indices or by
indexing an array (see [Constructors](@ref)) rather than calling it. The underlying
named-range model has [`NamedUnitRange`](@ref) as the named-range type that a tensor's
dimensions are ([`Index`](@ref) is the flavor keyed by an index name).

```@docs; canonical=false
AbstractNamedTensor
NamedTensor
NamedUnitRange
```

## Named array operations

Construct named objects with [`named`](@ref) and [`nameddims`](@ref), recover their parts
with [`name`](@ref), [`unnamed`](@ref), and [`dimnames`](@ref), and query their types with
[`dimnametype`](@ref), [`nametype`](@ref), and [`unnamedtype`](@ref). [`setname`](@ref) and
[`replacedimnames`](@ref) rename, and [`aligndims`](@ref) and [`aligneddims`](@ref) reorder a
tensor's dimensions by name (a copy and a view, respectively).

```@docs; canonical=false
named
nameddims
name
unnamed
dimnames
dimnametype
nametype
unnamedtype
setname
replacedimnames
aligndims
aligneddims
```

## Experimental

These features support building and applying operators, where an operator is a tensor whose
dimension names are split into an output set and an input set. The API is
still being refined and is subject to change. Build an operator with [`operator`](@ref) or
allocate one with [`similar_operator`](@ref), apply it to a tensor with [`apply`](@ref), and
recover its underlying tensor and name sets with [`state`](@ref), [`outputnames`](@ref),
and [`inputnames`](@ref).

```@docs; canonical=false
operator
similar_operator
apply
state
outputnames
inputnames
```
