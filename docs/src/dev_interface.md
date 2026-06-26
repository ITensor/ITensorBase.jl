# Developer Interface

```@meta
CurrentModule = ITensorBase
```

This page covers the named-array model that indices and tensors are built on, the interface
for implementing a new tensor type, and experimental features that are not yet part of the
stable user-facing API. For the stable user-facing API, see the [User Interface](@ref).

## Named array types

A concrete tensor type subtypes [`AbstractITensor`](@ref). [`NamedUnitRange`](@ref) is the
named-range type that a tensor's dimensions are ([`Index`](@ref) is the flavor keyed by an
index name).

```@docs; canonical=false
AbstractITensor
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
dimension names are split into a codomain (output) set and a domain (input) set. The API is
still being refined and is subject to change. Build an operator with [`operator`](@ref) or
allocate one with [`similar_operator`](@ref), apply it to a tensor with [`apply`](@ref), and
recover its underlying tensor and name sets with [`state`](@ref), [`codomainnames`](@ref),
and [`domainnames`](@ref).

```@docs; canonical=false
operator
similar_operator
apply
state
codomainnames
domainnames
```
