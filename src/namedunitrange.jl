using TensorAlgebra: TensorAlgebra, to_range, trivialrange, ungrade

"""
    NamedUnitRange{Name}

A unit range with a name attached, used as a named dimension (axis) of a tensor. It
pairs an underlying integer unit range with a name of type `Name`. [`Index`](@ref) is
the `NamedUnitRange` flavor whose name is an `IndexName`. Build one by calling
[`named`](@ref) on a range, or use `Index` to mint a fresh unique name.

# Examples

```jldoctest
julia> named(1:3, :i)
named(1:3, :i)
```

See also [`Index`](@ref), [`named`](@ref).
"""
struct NamedUnitRange{Name, UnnamedT, Unnamed} <: AbstractNamedVector{Name, UnnamedT}
    # The `value` is usually an integer `AbstractUnitRange`, but the bound is left open so a
    # backend can store a richer axis object directly, e.g. a native TensorKit space, which is
    # not an `AbstractUnitRange` but is its own axis (see the TensorKit extension).
    value::Unnamed
    name::Name
end

# Minimal interface.
unnamed(i::NamedUnitRange) = i.value
name(i::NamedUnitRange) = i.name
unnamedtype(::Type{<:NamedUnitRange{<:Any, <:Any, Unnamed}}) where {Unnamed} = Unnamed

# Construct from a space, minting a fresh name of the requested flavor. The space is
# anything `to_range` accepts (an `Integer`, an existing range, or a sector-pair vector
# when GradedArrays is loaded), so `Index(2)`, `Index(1:3)`, and
# `Index([U1(0) => 2, U1(1) => 3])` all work. Generic over the name type via `uniquename`,
# so `Index` (a `NamedUnitRange{IndexName}` alias) needs no `Index`-specific constructor.
function NamedUnitRange{Name}(space) where {Name}
    return NamedUnitRange{Name}(space, uniquename(Name))
end
# A space and an explicit name, name type fixed by the `Name` parameter: convert the space
# to a range, then fall through to the base case below.
function NamedUnitRange{Name}(space, name) where {Name}
    return NamedUnitRange{Name}(to_range(space), name)
end
# Base case: a ready-made range and a name. Fixing `Name` here rather than inferring it
# from `name` keeps the result a `NamedUnitRange{Name}` even if `uniquename(Name)` returns
# a different concrete type.
function NamedUnitRange{Name}(unnamed::AbstractUnitRange, name) where {Name}
    return NamedUnitRange{Name, eltype(unnamed), typeof(unnamed)}(unnamed, name)
end
# Base case for the name-inferred path: a ready-made range and a name. Loosening the struct's
# `Unnamed <: AbstractUnitRange` bound removed the compiler-synthesized 2-arg constructor that
# used to terminate here, so it is spelled out explicitly (a backend storing a non-range axis,
# e.g. a TensorKit space, adds its own terminal — see the TensorKit extension).
function NamedUnitRange(unnamed::AbstractUnitRange, name)
    return NamedUnitRange{typeof(name), eltype(unnamed), typeof(unnamed)}(unnamed, name)
end
# A space and an explicit name, name type inferred from `name`.
function NamedUnitRange(space, name)
    return NamedUnitRange(to_range(space), name)
end

# This can be customized to output different named unit range types.
namedunitrange(r::AbstractUnitRange, name) = NamedUnitRange(r, name)

# Mint a fresh trivial *named* range matching `r`'s backend: the trivial range of the
# underlying (unnamed) axis, carrying a fresh unique name of `r`'s name type.
function TensorAlgebra.trivialrange(r::NamedUnitRange{Name}) where {Name}
    return namedunitrange(trivialrange(unnamed(r)), uniquename(Name))
end
function TensorAlgebra.trivialrange(r::NamedUnitRange{Name}, n::Integer) where {Name}
    return namedunitrange(trivialrange(unnamed(r), n), uniquename(Name))
end

# Shorthand: attach an existing name to a range.
named(r::AbstractUnitRange, name) = namedunitrange(r, name)

# Derived interface. `setname` differs from the `AbstractNamedArray` method: it
# rebuilds through `named` so the result stays a named unit range, not a named
# array. The rest of the named interface (`isnamed`, `unnamedtype`, `nametype`,
# `uniquename`, `show`, `isempty`) is inherited from `AbstractNamedArray`; `==`,
# `isequal`, and `hash` are overridden just below.
# TODO: Use `Accessors.@set`?
setname(r::NamedUnitRange, name) = named(unnamed(r), name)

# Equality and hashing answer identity ("is this the same leg?"), keyed on the name plus the
# axis's ungraded extent (via `TensorAlgebra.ungrade`). Conjugation preserves the name and the
# ungraded extent while flipping arrows and charge labels, so an index compares equal to its
# dual and stock `Base` set-ops / `Dict` / `Set` treat the two as the same leg. `isequal`
# delegates to `==` (the Base default, valid since `==` returns a `Bool`), which also overrides
# the elementwise `AbstractArray` `isequal` that throws on a space-backed axis.
function Base.:(==)(r1::NamedUnitRange, r2::NamedUnitRange)
    return name(r1) == name(r2) && ungrade(unnamed(r1)) == ungrade(unnamed(r2))
end
Base.isequal(r1::NamedUnitRange, r2::NamedUnitRange) = r1 == r2
# `ungrade` strips the grading from the underlying range, keeping the name, so hashing runs
# through the shared `hash_named(:NamedArray, ...)` path on the ungraded range. That keeps `hash`
# consistent with `==` above and with a named array of equal values (Base's `[1, 2, 3] == 1:3`
# and `hash([1, 2, 3]) == hash(1:3)` contract).
TensorAlgebra.ungrade(r::NamedUnitRange) = named(ungrade(unnamed(r)), name(r))
Base.hash(r::NamedUnitRange, h::UInt) = hash_named(:NamedArray, ungrade(r), h)

# Forward `conj` to the underlying range so graded axes flip their sector
# arrows. The `Base.conj(::AbstractArray{<:Real}) = x` fallback would
# otherwise short-circuit before the inner range is touched.
Base.conj(r::NamedUnitRange) = named(conj(unnamed(r)), name(r))

# Unit range functionality.
Base.first(r::NamedUnitRange) = named(first(unnamed(r)), name(r))
Base.last(r::NamedUnitRange) = named(last(unnamed(r)), name(r))
# `length`, `size`, and `axes` are inherited from the `AbstractNamedArray` generic:
# the count and the positional axes are plain (unnamed). The element-layer methods
# (`first`, `last`, `step`, indexing, iteration) stay named.
Base.step(r::NamedUnitRange) = named(step(unnamed(r)), name(r))
Base.getindex(r::NamedUnitRange, I::Int) = getindex_named(r, I)
# Fix ambiguity error.
function Base.getindex(r::NamedUnitRange, I::AbstractUnitRange{<:Integer})
    return getindex_named(r, I)
end
# Fix ambiguity error.
function Base.getindex(r::NamedUnitRange, I::Colon)
    return getindex_named(r, I)
end
function Base.getindex(r::NamedUnitRange, I)
    return getindex_named(r, I)
end
# Fixes `r[begin]`/`r[end]`, since `firstindex` and `lastindex`
# returned named indices.
function Base.getindex(r::NamedUnitRange, I::NamedInteger)
    @assert name(r) == name(I)
    return getindex_named(r, unnamed(I))
end

# Named ranges are not `AbstractUnitRange`s, so `CartesianIndices` over a tuple of
# them has no Base method; unname to the parent ranges so `CartesianIndices` of a
# named tensor matches the parent's.
function Base.CartesianIndices(
        rs::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return CartesianIndices(unnamed.(rs))
end

# Show compactly; the inherited `AbstractNamedArray` text/plain show is multiline.
Base.show(io::IO, ::MIME"text/plain", r::NamedUnitRange) = show(io, r)

function Base.AbstractUnitRange{Int}(r::NamedUnitRange)
    return AbstractUnitRange{Int}(unnamed(r))
end

Base.oneto(length::NamedInteger) = named(Base.OneTo(unnamed(length)), name(length))
namedoneto(length::Integer, name) = Base.oneto(named(length, name))
Base.iterate(r::NamedUnitRange) = isempty(r) ? nothing : (first(r), first(r))
function Base.iterate(r::NamedUnitRange, i)
    i == last(r) && return nothing
    next = named(unnamed(i) + unnamed(step(r)), name(r))
    return (next, next)
end

struct NamedColon{Name} <: Function
    name::Name
end
unnamed(c::NamedColon) = Colon()
name(c::NamedColon) = c.name
named(::Colon, name) = NamedColon(name)

struct FirstIndex{Arr, Dim}
    array::Arr
    dim::Dim
end
Base.to_index(i::FirstIndex) = unnamed(first(axes(i.array, i.dim)))

struct LastIndex{Arr, Dim}
    array::Arr
    dim::Dim
end
Base.to_index(i::LastIndex) = unnamed(last(axes(i.array, i.dim)))

function Base.getindex(r::NamedUnitRange, I::FirstIndex)
    return first(r)
end
function Base.getindex(r::NamedUnitRange, I::LastIndex)
    return last(r)
end
