using LinearAlgebra: LinearAlgebra
using Random: Random
using TensorAlgebra: TensorAlgebra, permuteddims, zero!

# Some of the interface is inspired by:
# https://github.com/NamedTensor/ITensors.jl
# https://github.com/invenia/NamedDims.jl
# https://github.com/mcabbott/NamedPlus.jl
# https://pytorch.org/docs/stable/named_tensor.html

"""
    AbstractNamedTensor{DimName}

Supertype for tensors whose dimensions are labeled by names of type `DimName` rather
than ordered by position. Subtypes such as [`NamedTensor`](@ref) line their dimensions up
by name under contraction, addition, and indexing. Unlike an `AbstractArray`, the rank
and element type live in the data rather than the type, so `ndims` and `eltype` are not
fixed at the type level.

See also [`NamedTensor`](@ref), [`dimnames`](@ref), [`inds`](@ref).
"""
abstract type AbstractNamedTensor{DimName} end

# Rank and element type live in the data, not the type, so the type-level `ndims`
# is `Any` (like `eltype(Array)`). `AbstractNamedTensor` is not an `AbstractArray`: the
# array-like surface it needs (indexing, broadcasting, arithmetic, iteration) is
# supplied directly below rather than inherited.
Base.ndims(::Type{<:AbstractNamedTensor}) = Any

"""
    dimnames(a::AbstractNamedTensor)
    dimnames(a::AbstractNamedTensor, dim::Int)

The dimension names of `a`, as a collection in dimension order. The second form returns
the name of dimension `dim`.

# Examples

```jldoctest
julia> a = nameddims(zeros(2, 3), (:i, :j));

julia> dimnames(a)
2-element Vector{Symbol}:
 :i
 :j

julia> dimnames(a, 2)
:j
```

See also [`inds`](@ref), [`nameddims`](@ref).
"""
function dimnames end
dimnames(a::AbstractNamedTensor) = throw(MethodError(dimnames, a))
function dimnames(a::AbstractNamedTensor, dim::Int)
    return dimnames(a)[dim]
end

"""
    dimnametype(a::AbstractNamedTensor)
    dimnametype(type::Type{<:AbstractNamedTensor})

The type of an individual dimension name of `a`. The primary method dispatches
on the array type, and `dimnametype(a)` forwards to `dimnametype(typeof(a))`. A
type that does not fix its dimname flavor (such as the unparameterized `NamedTensor`)
returns `Any`, the same way `eltype(Array)` is `Any`.

# Examples

```jldoctest
julia> a = nameddims(zeros(2, 3), (:i, :j));

julia> dimnametype(a)
Symbol

julia> dimnametype(typeof(a))
Symbol
```
"""
function dimnametype end
dimnametype(a::AbstractNamedTensor) = dimnametype(typeof(a))
dimnametype(type::Type{<:AbstractNamedTensor}) = Any

# Unwrapping the names (named-array interface).
# TODO: Use `IsNamed` trait?
unnamed(a::AbstractNamedTensor) = throw(MethodError(unnamed, a))
function unnamed(a::AbstractNamedTensor, names)
    array = unnamed(a)
    # Fast path: already aligned to `names`, so no permutation and no `getperm` allocation.
    dimnames(a) == names && return array
    return permuteddims(array, Tuple(getperm(dimnames(a), names)))
end
unname(a::AbstractNamedTensor, inds) = unnamed(aligndims(a, inds))

"""
    inds(a::AbstractNamedTensor)
    inds(a::AbstractNamedTensor, dim::Int)

The named axes (indices) of `a`, as a `Vector` with one entry per dimension. Each entry
pairs a dimension's axis with its name. The second form returns the index of dimension
`dim`. Compare with [`dimnames`](@ref), which returns just the names without the axes. The
`axes` function returns the same indices as a `Tuple`, which the `AbstractArray` interface
relies on; `inds` returns a `Vector` because the indices are most often manipulated as a
collection (`filter`, `setdiff`, `union`).

# Examples

```jldoctest
julia> a = nameddims(zeros(2, 3), (:i, :j));

julia> inds(a)
2-element Vector{NamedUnitRange{Symbol, Int64, Base.OneTo{Int64}}}:
 named(Base.OneTo(2), :i)
 named(Base.OneTo(3), :j)

julia> inds(a, 1)
named(Base.OneTo(2), :i)
```
"""
function inds end
inds(a::AbstractNamedTensor) = collect(axes(a))
inds(a::AbstractNamedTensor, dim::Int) = axes(a)[dim]

isnamed(::Type{<:AbstractNamedTensor}) = true

function dim(a::AbstractNamedTensor, n)
    return findfirst(==(name(n)), dimnames(a))
end
dims(a::AbstractNamedTensor, ns) = Base.Fix1(dim, a).(ns)

dimname_isequal(x) = Base.Fix1(dimname_isequal, x)
dimname_isequal(x, y) = isequal(x, y)

dimname_isequal(r1::AbstractNamedArray, r2::AbstractNamedArray) = isequal(r1, r2)
dimname_isequal(r1::AbstractNamedArray, r2) = name(r1) == r2
dimname_isequal(r1, r2::AbstractNamedArray) = r1 == name(r2)

dimname_isequal(r1::AbstractNamedArray, r2::Name) = name(r1) == name(r2)
dimname_isequal(r1::Name, r2::AbstractNamedArray) = name(r1) == name(r2)

dimname_isequal(r1::NamedUnitRange, r2::NamedUnitRange) = isequal(r1, r2)
dimname_isequal(r1::NamedUnitRange, r2) = name(r1) == r2
dimname_isequal(r1, r2::NamedUnitRange) = r1 == name(r2)

dimname_isequal(r1::NamedUnitRange, r2::Name) = name(r1) == name(r2)
dimname_isequal(r1::Name, r2::NamedUnitRange) = name(r1) == name(r2)

function to_inds(a::AbstractNamedTensor, dims)
    is = Base.Fix1(dim, a).(name.(dims))
    return Base.Fix1(inds, a).(is)
end

# Generic construction of named dims arrays.

"""
    nameddims(a::AbstractArray, inds)

Construct a named dimensions array from an unnamed array `a` and named dimensions `inds`.

# Examples

```jldoctest
julia> nameddims(zeros(2, 3), (:i, :j))
named(Base.OneTo(2), :i)×named(Base.OneTo(3), :j) NamedTensor{Symbol}:
2×3 Matrix{Float64}:
 0.0  0.0  0.0
 0.0  0.0  0.0
```

See also [`NamedTensor`](@ref), [`named`](@ref).
"""
function nameddims(a::AbstractArray, inds)
    return NamedTensor(a, inds)
end

#=
    nameddimsof(a::AbstractNamedTensor, b::AbstractArray)

Construct a named dimensions array with the dimension names of `a`
and with the data from `b`.
=#
function nameddimsof(a::AbstractNamedTensor, b::AbstractArray)
    return nameddims(b, dimnames(a))
end

# TODO: Move to `utils.jl` file.
# TODO: Use `Base.indexin`?
function getperm(x, y; isequal = isequal)
    return map(yᵢ -> findfirst(isequal(yᵢ), x), y)
end

# TODO: Move to `utils.jl` file.
function checked_indexin(x, y)
    I = indexin(x, y)
    return something.(I)
end

function checked_indexin(x::Number, y)
    return findfirst(==(x), y)
end

function checked_indexin(x::AbstractUnitRange, y::AbstractUnitRange)
    return findfirst(==(first(x)), y):findfirst(==(last(x)), y)
end

Base.copy(a::AbstractNamedTensor) = nameddimsof(a, copy(unnamed(a)))
Base.zero(a::AbstractNamedTensor) = nameddimsof(a, zero(unnamed(a)))

# `CartesianIndices` of a named tensor is the parent's, via the named axes (as the
# `AbstractArray` fallback did through `axes`).
Base.CartesianIndices(a::AbstractNamedTensor) = CartesianIndices(axes(a))

# Forward `conj` to the underlying so that graded axes flip their sector arrows.
# The default `AbstractArray` fallback would broadcast `conj` over elements without
# touching the axes, which silently changes the contraction convention for tensors
# with graded (dual-tagged) axes.
Base.conj(a::AbstractNamedTensor) = nameddimsof(a, conj(unnamed(a)))

# `LinearAlgebra.normalize` infers result eltype via `typeof(first(a)/nrm)`, which
# scalar-indexes block-structured storage. `a / norm(a, p)` already preserves names.
function LinearAlgebra.normalize(a::AbstractNamedTensor, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end
function LinearAlgebra.normalize!(a::AbstractNamedTensor, p::Real = 2)
    LinearAlgebra.normalize!(unnamed(a), p)
    return a
end

# Elementwise and scalar arithmetic. `AbstractArray` routes these through
# broadcasting; supply them directly now that the supertype is gone.
Base.:+(a1::AbstractNamedTensor, a2::AbstractNamedTensor) = a1 .+ a2
Base.:-(a1::AbstractNamedTensor, a2::AbstractNamedTensor) = a1 .- a2
Base.:-(a::AbstractNamedTensor) = broadcast(-, a)
Base.:*(a::AbstractNamedTensor, x::Number) = a .* x
Base.:*(x::Number, a::AbstractNamedTensor) = x .* a
Base.:/(a::AbstractNamedTensor, x::Number) = a ./ x

# Forward `Random.randn!` / `Random.rand!` to the concrete storage so they
# see the runtime eltype.
function Random.randn!(rng::Random.AbstractRNG, a::AbstractNamedTensor)
    Random.randn!(rng, unnamed(a))
    return a
end
function Random.rand!(rng::Random.AbstractRNG, a::AbstractNamedTensor)
    Random.rand!(rng, unnamed(a))
    return a
end

function Base.copyto!(a_dest::AbstractNamedTensor, a_src::AbstractNamedTensor)
    a′_dest = unnamed(a_dest)
    # TODO: Use `unnamed` to do the permutations lazily.
    a′_src = unname(a_src, axes(a_dest))
    copyto!(a′_dest, a′_src)
    return a_dest
end

# Conversion

# Copied from `Base` (defined in abstractarray.jl).
@noinline function _checkaxs(axd, axs)
    axd == axs || throw(DimensionMismatch("axes must agree, got $axd and $axs"))
    return nothing
end
function copyto_axcheck!(dest, src)
    _checkaxs(axes(dest), axes(src))
    copyto!(dest, src)
    return dest
end

# These are defined since the Base versions assume the eltype and ndims are known
# at compile time, which isn't true for ITensors.
Base.Array(a::AbstractNamedTensor) = Array(unnamed(a))
Base.Array{T}(a::AbstractNamedTensor) where {T} = Array{T}(unnamed(a))
Base.Array{T, N}(a::AbstractNamedTensor) where {T, N} = Array{T, N}(unnamed(a))
Base.AbstractArray{T}(a::AbstractNamedTensor) where {T} = AbstractArray{T, ndims(a)}(a)
function Base.AbstractArray{T, N}(a::AbstractNamedTensor) where {T, N}
    dest = similar(a, T)
    copyto_axcheck!(unnamed(dest), unnamed(a))
    return dest
end

function Base.axes(a::AbstractNamedTensor)
    return named.(axes(unnamed(a)), Tuple(dimnames(a)))
end
function Base.size(a::AbstractNamedTensor)
    return length.(axes(a))
end

# An NamedTensor has no single name, so `length` is the plain element count. It is the
# product of the (now plain `Int`) per-dimension sizes.
Base.length(a::AbstractNamedTensor) = prod(size(a))

# Circumvent issue when ndims isn't known at compile time.
Base.axes(a::AbstractNamedTensor, d) = axes(a)[d]

# Circumvent issue when ndims isn't known at compile time.
Base.size(a::AbstractNamedTensor, d) = size(a)[d]

# Circumvent issue when ndims isn't known at compile time.
Base.ndims(a::AbstractNamedTensor) = ndims(unnamed(a))

# Circumvent issue when eltype isn't known at compile time.
Base.eltype(a::AbstractNamedTensor) = eltype(unnamed(a))

# In-place zero of an NamedTensor, delegating to its unnamed parent array.
TensorAlgebra.zero!(a::AbstractNamedTensor) = (zero!(unnamed(a)); a)

# Name-aware `VectorInterface` methods so that ITensors can be used directly as the vectors
# in iterative solvers such as `KrylovKit.eigsolve`, which drive their Krylov vectors through
# `VectorInterface`; the generic `AbstractArray` fallbacks are not name-aware. The `!` methods
# operate in place via broadcasting; each `!!` method does so too when the result fits the
# destination's element type, and otherwise allocates. `scalartype` is computed in the value
# domain because an NamedTensor's element type is not encoded in its type.
using VectorInterface: VectorInterface, add, add!, scalartype, scale, scale!, zerovector!
VectorInterface.scalartype(a::AbstractNamedTensor) = scalartype(unnamed(a))
function VectorInterface.scalartype(a::AbstractArray{<:AbstractNamedTensor})
    return mapreduce(scalartype, promote_type, a; init = Bool)
end

function VectorInterface.zerovector(
        a::AbstractNamedTensor,
        ::Type{S}
    ) where {S <: Number}
    return zerovector!(similar(a, S))
end
VectorInterface.zerovector!(a::AbstractNamedTensor) = zero!(a)
VectorInterface.zerovector!!(a::AbstractNamedTensor) = zerovector!(a)

VectorInterface.scale(a::AbstractNamedTensor, α::Number) = a * α
function VectorInterface.scale!(a::AbstractNamedTensor, α::Number)
    @. a = a * α
    return a
end
function VectorInterface.scale!(
        b::AbstractNamedTensor,
        a::AbstractNamedTensor,
        α::Number
    )
    @. b = a * α
    return b
end
function VectorInterface.scale!!(a::AbstractNamedTensor, α::Number)
    promote_type(scalartype(a), typeof(α)) <: scalartype(a) || return scale(a, α)
    return scale!(a, α)
end
function VectorInterface.scale!!(
        b::AbstractNamedTensor,
        a::AbstractNamedTensor,
        α::Number
    )
    promote_type(scalartype(b), scalartype(a), typeof(α)) <: scalartype(b) ||
        return scale(a, α)
    return scale!(b, a, α)
end

function VectorInterface.add(
        y::AbstractNamedTensor,
        x::AbstractNamedTensor,
        α::Number,
        β::Number
    )
    return @. y * β + x * α
end
function VectorInterface.add!(
        y::AbstractNamedTensor,
        x::AbstractNamedTensor,
        α::Number,
        β::Number
    )
    @. y = y * β + x * α
    return y
end
function VectorInterface.add!!(
        y::AbstractNamedTensor,
        x::AbstractNamedTensor,
        α::Number,
        β::Number
    )
    promote_type(scalartype(y), scalartype(x), typeof(α), typeof(β)) <: scalartype(y) ||
        return add(y, x, α, β)
    return add!(y, x, α, β)
end

function VectorInterface.inner(x::AbstractNamedTensor, y::AbstractNamedTensor)
    return (conj(x) * y)[]
end

Base.axes(a::AbstractNamedTensor, dimname::Name) = axes(a, dim(a, dimname))
Base.size(a::AbstractNamedTensor, dimname::Name) = size(a, dim(a, dimname))

function similar_nameddims(a::AbstractNamedTensor, elt::Type, ax)
    return nameddims(
        similar(unnamed(a), elt, unnamed.(Tuple(ax))),
        name.(ax)
    )
end
function similar_nameddims(a::AbstractArray, elt::Type, ax)
    return nameddims(similar(a, elt, unnamed.(Tuple(ax))), name.(ax))
end

# Base.similar gets the eltype at compile time.
Base.similar(a::AbstractNamedTensor) = similar(a, eltype(a))
function Base.similar(a::AbstractNamedTensor, elt::Type)
    return similar_nameddims(a, elt)
end
function similar_nameddims(a::AbstractNamedTensor, elt::Type)
    return nameddimsof(a, similar(unnamed(a), elt))
end

# This is defined explicitly since the Base version expects the eltype
# to be known at compile time, which isn't true for ITensors.
function Base.similar(
        a::AbstractArray, inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return similar(a, eltype(a), inds)
end

function Base.similar(
        a::AbstractArray, elt::Type,
        inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return similar_nameddims(a, elt, inds)
end

# Same entry points with a named-tensor prototype. An `AbstractNamedTensor` is no longer
# an `AbstractArray`, so the methods above (which build a named tensor from a plain
# array prototype) no longer cover it.
function Base.similar(
        a::AbstractNamedTensor,
        inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return similar(a, eltype(a), inds)
end
function Base.similar(
        a::AbstractNamedTensor, elt::Type,
        inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return similar_nameddims(a, elt, inds)
end
function setdimnames(a::AbstractNamedTensor, dimnames)
    return nameddims(unnamed(a), dimnames)
end

"""
    replacedimnames(a::AbstractNamedTensor, replacements::Pair...)
    replacedimnames(f, a::AbstractNamedTensor)

Return a tensor with the same data as `a` but with its dimension names replaced. The
first form takes `old => new` pairs, replacing matching names and leaving the rest
unchanged. The second form replaces each name with `f(name)`.

# Examples

```jldoctest
julia> using ITensorBase: replacedimnames

julia> a = nameddims(zeros(2, 3), (:i, :j));

julia> dimnames(replacedimnames(a, :i => :k))
2-element Vector{Symbol}:
 :k
 :j
```

See also [`dimnames`](@ref).
"""
function replacedimnames end
function replacedimnames(a::AbstractNamedTensor, replacements::Pair...)
    new_dimnames = replace(dimnames(a), replacements...)
    return nameddims(unnamed(a), new_dimnames)
end
function replacedimnames(f, a::AbstractNamedTensor)
    new_dimnames = replace(f, dimnames(a))
    return nameddims(unnamed(a), new_dimnames)
end
mapdimnames(f, a::AbstractNamedTensor) = replacedimnames(f, a)

# Replace over `axes` (a `Tuple`) rather than `inds` (a `Vector`): `replace` on a `Vector`
# is homogeneous and would fail to convert a replacement index backed by a different range
# type (e.g. `UnitRange` into a `OneTo`-backed vector), whereas a `Tuple` admits the mixed
# element types. The result is splatted into `getindex`, so only the order matters.
function replaceinds(a::AbstractNamedTensor, replacements::Pair...)
    new_inds = replace(axes(a), replacements...)
    return unnamed(a)[new_inds...]
end
function replaceinds(f, a::AbstractNamedTensor)
    new_inds = replace(f, axes(a))
    return unnamed(a)[new_inds...]
end
mapinds(f, a::AbstractNamedTensor) = replaceinds(f, a)

# `Base.isempty(a::AbstractArray)` is defined as `length(a) == 0`,
# which involves comparing a named integer to an unnamed integer
# which isn't well defined.
Base.isempty(a::AbstractNamedTensor) = isempty(unnamed(a))

# Define this on objects rather than types in case the wrapper type
# isn't known at compile time, like for the NamedTensor type.
Base.IndexStyle(a::AbstractNamedTensor) = IndexStyle(unnamed(a))
Base.eachindex(a::AbstractNamedTensor) = eachindex(unnamed(a))

# Iteration, keys, and pairs forward to the parent (these were previously inherited
# from `AbstractArray`).
Base.iterate(a::AbstractNamedTensor, state...) = iterate(unnamed(a), state...)
Base.keys(a::AbstractNamedTensor) = keys(unnamed(a))
Base.pairs(a::AbstractNamedTensor) = pairs(unnamed(a))

# Multi-argument `eachindex` dispatches on the named index style, as the
# `AbstractArray` version did.
function Base.eachindex(a1::AbstractNamedTensor, a_rest::AbstractNamedTensor...)
    return eachindex(IndexStyle(a1, a_rest...), a1, a_rest...)
end

# Cartesian indices with names attached.
struct NamedIndexCartesian <: IndexStyle end

# When multiple named dims arrays are involved, use the named
# dimensions.
function Base.IndexStyle(a1::AbstractNamedTensor, a2::AbstractNamedTensor)
    return NamedIndexCartesian()
end
# Define promotion of index styles.
Base.IndexStyle(s1::NamedIndexCartesian, s2::NamedIndexCartesian) = NamedIndexCartesian()
Base.IndexStyle(s1::IndexStyle, s2::NamedIndexCartesian) = NamedIndexCartesian()
Base.IndexStyle(s1::NamedIndexCartesian, s2::IndexStyle) = NamedIndexCartesian()

# Like CartesianIndex but with named dimensions.
struct NamedDimsCartesianIndex{N, Index <: Tuple{Vararg{NamedInteger, N}}} <:
    Base.AbstractCartesianIndex{N}
    I::Index
end
NamedDimsCartesianIndex(I::NamedInteger...) = NamedDimsCartesianIndex(I)
Base.Tuple(I::NamedDimsCartesianIndex) = I.I
function Base.show(io::IO, I::NamedDimsCartesianIndex)
    print(io, "NamedDimsCartesianIndex")
    show(io, Tuple(I))
    return nothing
end

# Like CartesianIndices but with named dimensions.
struct NamedDimsCartesianIndices{
        N,
        DimName,
        Indices <: Tuple{Vararg{NamedUnitRange, N}},
        Index <: Tuple{Vararg{NamedInteger, N}},
    } <: AbstractNamedTensor{DimName}
    indices::Indices
    function NamedDimsCartesianIndices(indices::Tuple{Vararg{NamedUnitRange}})
        dimname = eltype(name.(indices))
        return new{length(indices), dimname, typeof(indices), Tuple{eltype.(indices)...}}(
            indices
        )
    end
end
# The element type is no longer carried by the (rank-erased) supertype, so recover
# it from the stored index-tuple parameter.
function Base.eltype(
        ::Type{<:NamedDimsCartesianIndices{N, <:Any, <:Any, Index}}
    ) where {N, Index}
    return NamedDimsCartesianIndex{N, Index}
end
Base.eltype(I::NamedDimsCartesianIndices) = eltype(typeof(I))
Base.axes(I::NamedDimsCartesianIndices) = (only ∘ axes).(I.indices)
Base.size(I::NamedDimsCartesianIndices) = length.(I.indices)

function Base.getindex(a::NamedDimsCartesianIndices{N}, I::Vararg{Int, N}) where {N}
    index = map(a.indices, I) do r, i
        return r[i]
    end
    return NamedDimsCartesianIndex(index)
end

function unnamed(I::NamedDimsCartesianIndices)
    return CartesianIndices(unnamed.(I.indices))
end

# Iterating yields `NamedDimsCartesianIndex`es. The generic `AbstractNamedTensor`
# iteration forwards to `unnamed`, which here is a plain `CartesianIndices`, so
# convert each parent index back through `getindex`.
function Base.iterate(I::NamedDimsCartesianIndices, state...)
    y = iterate(unnamed(I), state...)
    isnothing(y) && return nothing
    cartesian, next_state = y
    return I[Tuple(cartesian)...], next_state
end

function Base.eachindex(
        ::NamedIndexCartesian,
        a1::AbstractNamedTensor,
        a_rest::AbstractNamedTensor...
    )
    all(a -> issetequal(dimnames(a1), dimnames(a)), a_rest) ||
        throw(NameMismatch("Dimension name mismatch $(dimnames.((a1, a_rest...)))."))
    # TODO: Check the shapes match.
    return NamedDimsCartesianIndices(axes(a1))
end

# `unname` (eager), not `unnamed` (lazy view): reducing over a lazy permuted view
# scalar-indexes, which graded arrays forbid.

# Base version ignores dimension names.
# TODO: Use `mapreduce(isequal, &&, a1, a2)`?
function Base.isequal(a1::AbstractNamedTensor, a2::AbstractNamedTensor)
    issetequal(dimnames(a1), dimnames(a2)) || return false
    return isequal(unnamed(a1), unname(a2, dimnames(a1)))
end

# Base version ignores dimension names.
# TODO: Use `mapreduce(==, &&, a1, a2)`?
# TODO: Handle `missing` values properly.
function Base.:(==)(a1::AbstractNamedTensor, a2::AbstractNamedTensor)
    issetequal(dimnames(a1), dimnames(a2)) || return false
    return unnamed(a1) == unname(a2, dimnames(a1))
end

# Base version ignores dimension names.
function Base.isapprox(a1::AbstractNamedTensor, a2::AbstractNamedTensor; kwargs...)
    issetequal(dimnames(a1), dimnames(a2)) || return false
    return isapprox(unnamed(a1), unname(a2, dimnames(a1)); kwargs...)
end

# Generalization of `Base.sort` to Tuples for Julia v1.10 compatibility.
# TODO: Remove when we drop support for Julia v1.10.
_sort(x; kwargs...) = sort(x; kwargs...)
_sort(x::NTuple{N}; kwargs...) where {N} = NTuple{N}(sort(collect(x); kwargs...))

function Base.hash(a::AbstractNamedTensor, h::UInt64)
    h = hash(:NamedTensor, h)
    a′ = aligneddims(a, _sort(dimnames(a)))
    h = hash(unnamed(a′), h)
    for i in axes(a′)
        h = hash(i, h)
    end
    return h
end

# Indexing.

# Scalar indexing

Base.firstindex(a::AbstractNamedTensor) = firstindex(unnamed(a))
Base.lastindex(a::AbstractNamedTensor) = lastindex(unnamed(a))

function Base.firstindex(a::AbstractNamedTensor, d)
    return FirstIndex(a, d)
end

function Base.lastindex(a::AbstractNamedTensor, d)
    return LastIndex(a, d)
end

# Redefine generic definition which expects `axes(a)` to
# return a Tuple.
function Base.to_indices(a::AbstractNamedTensor, I::Tuple)
    return to_indices(a, Tuple(axes(a)), I)
end
# Fix ambiguity error with Base.
function Base.to_indices(
        a::AbstractNamedTensor,
        I::Tuple{Union{Integer, CartesianIndex}}
    )
    return to_indices(a, Tuple(axes(a)), I)
end
function Base.checkbounds(::Type{Bool}, a::AbstractNamedTensor, I::Int...)
    return checkbounds(Bool, unnamed(a), I...)
end

function Base.to_indices(
        a::AbstractNamedTensor, I::Tuple{NamedInteger, Vararg{NamedInteger}}
    )
    perm = getperm(name.(I), dimnames(a))
    # TODO: Throw a `NameMismatch` error.
    @assert isperm(perm)
    I = map(p -> I[p], perm)
    return map(axes(a), I) do dimname, i
        return checked_indexin(unnamed(i), unnamed(dimname))
    end
end
function Base.to_indices(
        a::AbstractNamedTensor, I::Tuple{Pair{<:Any, Int}, Vararg{Pair{<:Any, Int}}}
    )
    inds = to_inds(a, first.(I))
    return to_indices(a, map((i, name) -> name[i], last.(I), inds))
end
function Base.to_indices(a::AbstractNamedTensor, I::Tuple{Pair, Vararg{Pair}})
    inds = to_inds(a, first.(I))
    return map((i, name) -> name[i], last.(I), inds)
    return to_indices(a, named.(last.(I), first.(I)))
end

function Base.to_indices(a::AbstractNamedTensor, I::Tuple{NamedDimsCartesianIndex})
    return to_indices(a, Tuple(only(I)))
end

function Base.getindex(a::AbstractNamedTensor, I...)
    return getindex(a, to_indices(a, I)...)
end

function Base.getindex(a::AbstractNamedTensor, I1::Int, Irest::Int...)
    return getindex(unnamed(a), I1, Irest...)
end
function Base.getindex(
        a::AbstractNamedTensor, I1::NamedInteger, Irest::NamedInteger...
    )
    return getindex(a, to_indices(a, (I1, Irest...))...)
end
function Base.getindex(a::AbstractNamedTensor)
    return getindex(unnamed(a))
end
# Linear indexing.
function Base.getindex(a::AbstractNamedTensor, I::Int)
    return getindex(unnamed(a), I)
end

function Base.setindex!(a::AbstractNamedTensor, value, I1::Int, Irest::Int...)
    setindex!(unnamed(a), value, I1, Irest...)
    return a
end
function Base.setindex!(a::AbstractNamedTensor, value, I::CartesianIndex)
    setindex!(a, value, to_indices(a, (I,))...)
    return a
end

function Base.setindex!(
        a::AbstractNamedTensor, value, I1::NamedInteger,
        Irest::NamedInteger...
    )
    setindex!(a, value, to_indices(a, (I1, Irest...))...)
    return a
end
function Base.setindex!(a::AbstractNamedTensor, value, I::NamedDimsCartesianIndex)
    setindex!(a, value, to_indices(a, (I,))...)
    return a
end
function Base.setindex!(a::AbstractNamedTensor, value, I1::Pair, Irest::Pair...)
    setindex!(a, value, to_indices(a, (I1, Irest...))...)
    return a
end
function Base.setindex!(a::AbstractNamedTensor, value)
    setindex!(unnamed(a), value)
    return a
end
# Linear indexing.
function Base.setindex!(a::AbstractNamedTensor, value, I::Int)
    setindex!(unnamed(a), value, I)
    return a
end

function Base.isassigned(a::AbstractNamedTensor, I::Int...)
    return isassigned(unnamed(a), I...)
end

# Slicing

# Like `const ViewIndex = Union{Real,AbstractArray}`.
const NamedViewIndex =
    Union{NamedInteger, NamedUnitRange, AbstractNamedArray}

using ArrayLayouts: ArrayLayouts, MemoryLayout

# Parent array type. Methods are defined per concrete type (`NamedTensor`,
# `NamedTensorOperator`); declared here since `MemoryLayout` below dispatches on it.
function parenttype end

abstract type AbstractNamedTensorLayout <: MemoryLayout end
struct NamedTensorLayout{ParentLayout} <: AbstractNamedTensorLayout end

function ArrayLayouts.MemoryLayout(arrtype::Type{<:AbstractNamedTensor})
    return NamedTensorLayout{typeof(MemoryLayout(parenttype(arrtype)))}()
end

function ArrayLayouts.sub_materialize(::NamedTensorLayout, a, ax)
    return copy(a)
end

# TODO: Should this be a view?
function Base.getindex(a::AbstractArray, I1::Name, Irest::Name...)
    return copy(view(a, I1, Irest...))
end
function Base.view(a::AbstractArray, I1::Name, Irest::Name...)
    return nameddims(a, name.((I1, Irest...)))
end

function Base.getindex(a::AbstractArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
    return copy(view(a, I1, Irest...))
end
# A named unit range is an `AbstractArray`, so for a concrete `Array` the Base
# `getindex(::Array, ::AbstractVector)` method would otherwise win over the generic
# named `getindex` above. This restores the named behavior for `Array`.
function Base.getindex(a::Array, I1::NamedUnitRange)
    return copy(view(a, I1))
end
function Base.view(a::AbstractArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
    I = (I1, Irest...)
    return nameddims(view(a, unnamed.(I)...), name.(I))
end

# TODO: Should this be a view?
function Base.getindex(a::AbstractNamedTensor, I1::Name, Irest::Name...)
    return copy(view(a, I1, Irest...))
end
function Base.view(a::AbstractNamedTensor, I1::Name, Irest::Name...)
    issetequal(dimnames(a), name.((I1, Irest...))) ||
        throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(name.((I1, Irest...)))."
        )
    )
    return a
end

function Base.getindex(a::AbstractNamedTensor, I1::Pair, Irest::Pair...)
    return getindex(a, to_indices(a, (I1, Irest...))...)
end
function Base.view(a::AbstractNamedTensor, I1::Pair, Irest::Pair...)
    I = (I1, Irest...)
    inds = to_inds(a, first.(I))
    return view(a, map((i, name) -> name[i], last.(I), inds)...)
end

function Base.getindex(
        a::AbstractNamedTensor, I1::NamedViewIndex, Irest::NamedViewIndex...
    )
    return copy(view(a, I1, Irest...))
end
function Base.view(a::AbstractNamedTensor, I1::NamedViewIndex, Irest::NamedViewIndex...)
    I = (I1, Irest...)
    perm = getperm(name.(I), dimnames(a))
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(name.(I))."
        )
    )
    Ip = map(p -> unnamed(I[p]), perm)
    return view_nameddims(a, Ip...)
end

# Repeated definition of `Base.ViewIndex`.
const ViewIndex = Union{Real, AbstractArray}

# Like `Base.ScalarIndex` but as a trait function.
# This catches cases like `Colon`, `BlockArrays.Block`, etc. which are not AbstractArray
# indices but also aren't scalar indices.
isscalarindex(I) = false
isscalarindex(I::Real) = true

# Slicing with unnamed indices, such as:
# a = NamedTensor(rand(3,4), (:x, :y))
# b = view(a, 1:2, 2)
function view_nameddims(a::AbstractNamedTensor, I...)
    nonscalar_dims = filter(dim -> !isscalarindex(I[dim]), ntuple(identity, ndims(a)))
    nonscalar_dimnames = map(dim -> dimnames(a, dim), nonscalar_dims)
    return nameddims(view(unnamed(a), I...), nonscalar_dimnames)
end

function Base.view(a::AbstractNamedTensor, I::ViewIndex...)
    return view_nameddims(a, I...)
end

function getindex_nameddims(a::AbstractNamedTensor, I...)
    return copy(view(a, I...))
end

function Base.getindex(a::AbstractNamedTensor, I::ViewIndex...)
    return getindex_nameddims(a, I...)
end

function Base.setindex!(
        a::AbstractNamedTensor,
        value::AbstractNamedTensor,
        I1::NamedViewIndex,
        Irest::NamedViewIndex...
    )
    view(a, I1, Irest...) .= value
    return a
end
function Base.setindex!(
        a::AbstractNamedTensor,
        value::AbstractArray,
        I1::NamedViewIndex,
        Irest::NamedViewIndex...
    )
    I = (I1, Irest...)
    a[I...] = nameddims(value, name.(I))
    return a
end
function Base.setindex!(
        a::AbstractNamedTensor,
        value::AbstractNamedTensor,
        I1::ViewIndex,
        Irest::ViewIndex...
    )
    view(a, I1, Irest...) .= value
    return a
end
function Base.setindex!(
        a::AbstractNamedTensor, value::AbstractArray, I1::ViewIndex, Irest::ViewIndex...
    )
    setindex!(unnamed(a), value, I1, Irest...)
    return a
end

# Permute/align dimensions

"""
    aligndims(a::AbstractNamedTensor, dims)

Reorder the dimensions of `a` into the order given by `dims`, matched by name. Returns a
tensor with the same data and dimension names as `a` but with the dimensions permuted, and
throws a `NameMismatch` if `dims` is not a permutation of `a`'s dimension names.

# Examples

```jldoctest
julia> a = nameddims(zeros(2, 3), (:i, :j));

julia> aligndims(a, (:j, :i))
named(Base.OneTo(3), :j)×named(Base.OneTo(2), :i) NamedTensor{Symbol}:
3×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0
 0.0  0.0
```
"""
function aligndims(a::AbstractNamedTensor, dims)
    new_dimnames = name.(dims)
    perm = Tuple(getperm(dimnames(a), new_dimnames))
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(new_dimnames)."
        )
    )
    return nameddims(permutedims(unnamed(a), perm), new_dimnames)
end

"""
    aligneddims(a::AbstractNamedTensor, dims)

Like [`aligndims`](@ref), but returns a lazily-permuted view that shares data with `a`
instead of copying. Reorders the dimensions of `a` into the order given by `dims`, matched by
name, and throws a `NameMismatch` if `dims` is not a permutation of `a`'s dimension names.

# Examples

```jldoctest
julia> a = nameddims(reshape(1:6, 2, 3), (:i, :j));

julia> dimnames(aligneddims(a, (:j, :i)))
2-element Vector{Symbol}:
 :j
 :i
```

See also [`aligndims`](@ref).
"""
function aligneddims(a::AbstractNamedTensor, dims)
    new_dimnames = name.(dims)
    perm = getperm(dimnames(a), new_dimnames)
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(new_dimnames)."
        )
    )
    return nameddims(
        permuteddims(unnamed(a), perm), new_dimnames
    )
end

# Convenient constructors

using Random: Random, AbstractRNG

# Like `Base.rand` but supports axes, not just size.
# TODO: Come up with a better name for this.
_rand(args...) = Base.rand(args...)
function _rand(
        rng::AbstractRNG, elt::Type, dims::Tuple{Base.OneTo{Int}, Vararg{Base.OneTo{Int}}}
    )
    return Base.rand(rng, elt, length.(dims))
end

# Like `Base.randn` but supports axes, not just size.
# TODO: Come up with a better name for this.
_randn(args...) = Base.randn(args...)
function _randn(
        rng::AbstractRNG, elt::Type, dims::Tuple{Base.OneTo{Int}, Vararg{Base.OneTo{Int}}}
    )
    return Base.randn(rng, elt, length.(dims))
end

default_eltype() = Float64
for (f, f′) in [(:rand, :_rand), (:randn, :_randn)]
    @eval begin
        function Base.$f(
                rng::AbstractRNG,
                elt::Type{<:Number},
                ax::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
            )
            a = $f′(rng, elt, unnamed.(ax))
            return a[Name.(name.(ax))...]
        end
        function Base.$f(
                rng::AbstractRNG,
                elt::Type{<:Number},
                dims::Tuple{NamedInteger, Vararg{NamedInteger}}
            )
            return $f(rng, elt, Base.oneto.(dims))
        end
    end
    for dimtype in [:NamedInteger, :NamedUnitRange]
        @eval begin
            function Base.$f(
                    rng::AbstractRNG, elt::Type{<:Number}, dim1::$dimtype,
                    dims::Vararg{$dimtype}
                )
                return $f(rng, elt, (dim1, dims...))
            end
            Base.$f(elt::Type{<:Number}, dims::Tuple{$dimtype, Vararg{$dimtype}}) = $f(
                Random.default_rng(), elt, dims
            )
            Base.$f(elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype}) = $f(
                elt, (dim1, dims...)
            )
            Base.$f(dims::Tuple{$dimtype, Vararg{$dimtype}}) = $f(default_eltype(), dims)
            Base.$f(dim1::$dimtype, dims::Vararg{$dimtype}) = $f((dim1, dims...))
        end
    end
end
for f in [:zeros, :ones], dimtype in [:NamedInteger, :NamedUnitRange]
    @eval begin
        function Base.$f(
                elt::Type{<:Number}, ax::Tuple{$dimtype, Vararg{$dimtype}}
            )
            a = $f(elt, unnamed.(ax))
            return a[Name.(name.(ax))...]
        end
        function Base.$f(elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype})
            return $f(elt, (dim1, dims...))
        end
        Base.$f(dims::Tuple{$dimtype, Vararg{$dimtype}}) = $f(default_eltype(), dims)
        Base.$f(dim1::$dimtype, dims::Vararg{$dimtype}) = $f((dim1, dims...))
    end
end
for dimtype in [:NamedInteger, :NamedUnitRange]
    @eval begin
        function Base.fill(value, ax::Tuple{$dimtype, Vararg{$dimtype}})
            a = fill(value, unnamed.(ax))
            return a[Name.(name.(ax))...]
        end
        function Base.fill(value, dim1::$dimtype, dims::Vararg{$dimtype})
            return fill(value, (dim1, dims...))
        end
    end
end

function Base.fill!(a::AbstractNamedTensor, v)
    fill!(unnamed(a), v)
    return a
end

function Base.map!(f, a_dest::AbstractNamedTensor, a_srcs::AbstractNamedTensor...)
    a′_dest = unnamed(a_dest)
    # TODO: Use `unnamed` to do the permutations lazily.
    # TODO: Define `unname[d](dimnames) = Base.Fix1(unname[d], dimnames)` and use it here?
    a′_srcs = Base.Fix2(unname, dimnames(a_dest)).(a_srcs)
    map!(f, a′_dest, a′_srcs...)
    return a_dest
end

function Base.map(f, a_srcs::AbstractNamedTensor...)
    # copy(mapped(f, a_srcs...))
    return f.(a_srcs...)
end

function Base.mapreduce(f, op, a::AbstractNamedTensor; kwargs...)
    return mapreduce(f, op, unnamed(a); kwargs...)
end

# `sum` is routed to the underlying data rather than left to fall back on the
# `mapreduce` method above because some array types (such as graded arrays) define
# `Base.sum` directly but not the general `mapreduce`, so the unnamed `sum` is the
# path that works for them.
function Base.sum(a::AbstractNamedTensor; kwargs...)
    return sum(unnamed(a); kwargs...)
end

function LinearAlgebra.promote_leaf_eltypes(a::AbstractNamedTensor)
    return LinearAlgebra.promote_leaf_eltypes(unnamed(a))
end

# Printing

# Copy of `Base.dims2string` defined in `show.jl`.
function dims_to_string(d)
    isempty(d) && return "0-dimensional"
    length(d) == 1 && return "$(d[1])-element"
    return join(map(string, d), '×')
end

function concretetype_to_string_truncated(
        type::Type;
        param_truncation_length = typemax(Int)
    )
    isconcretetype(type) || throw(ArgumentError("Type must be concrete."))
    alias = Base.make_typealias(type)
    base_type, params = if isnothing(alias)
        Base.typename(type).wrapper, type.parameters
    else
        base_type_globalref, params_svec = alias
        base_type_globalref.name, params_svec
    end
    str = string(base_type)
    if isempty(params)
        return str
    end
    str *= '{'
    param_strings = map(params) do param
        param_string = string(param)
        if length(param_string) > param_truncation_length
            return "…"
        end
        return param_string
    end
    str *= join(param_strings, ", ")
    str *= '}'
    return str
end

function Base.summary(io::IO, a::AbstractNamedTensor)
    print(io, dims_to_string(inds(a)))
    print(io, ' ')
    print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length = 40))
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", a::AbstractNamedTensor)
    summary(io, a)
    println(io, ":")
    show(io, mime, unnamed(a))
    return nothing
end

function Base.show(io::IO, a::AbstractNamedTensor)
    show(io, unnamed(a))
    print(io, "[", join(inds(a), ", "), "]")
    return nothing
end
