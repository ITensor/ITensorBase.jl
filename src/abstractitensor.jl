using LinearAlgebra: LinearAlgebra
using Random: Random
using TensorAlgebra: permuteddims
using TypeParameterAccessors: unspecify_type_parameters

# Some of the interface is inspired by:
# https://github.com/ITensor/ITensors.jl
# https://github.com/invenia/NamedDims.jl
# https://github.com/mcabbott/NamedPlus.jl
# https://pytorch.org/docs/stable/named_tensor.html

abstract type AbstractITensor{DimName} end

# Rank and element type live in the data, not the type, so the type-level `ndims`
# is `Any` (like `eltype(Array)`). `AbstractITensor` is not an `AbstractArray`: the
# array-like surface it needs (indexing, broadcasting, arithmetic, iteration) is
# supplied directly below rather than inherited.
Base.ndims(::Type{<:AbstractITensor}) = Any

dimnames(a::AbstractITensor) = throw(MethodError(dimnames, a))
function dimnames(a::AbstractITensor, dim::Int)
    return dimnames(a)[dim]
end

"""
    dimnametype(a::AbstractITensor)
    dimnametype(type::Type{<:AbstractITensor})

The type of an individual dimension name of `a`. The primary method dispatches
on the array type, and `dimnametype(a)` forwards to `dimnametype(typeof(a))`. A
type that does not fix its dimname flavor (such as the unparameterized `ITensor`)
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
dimnametype(a::AbstractITensor) = dimnametype(typeof(a))
dimnametype(type::Type{<:AbstractITensor}) = Any

# Unwrapping the names (named-array interface).
# TODO: Use `IsNamed` trait?
denamed(a::AbstractITensor) = throw(MethodError(denamed, a))
denamed(a::AbstractITensor, inds) = denamed(aligneddims(a, inds))
dename(a::AbstractITensor, inds) = denamed(aligndims(a, inds))

# Output the named axes/indices of the named dims array, as a `Tuple` (even though
# the dimension names are stored as a `Vector`).
inds(a::AbstractITensor) = named.(axes(denamed(a)), Tuple(dimnames(a)))
inds(a::AbstractITensor, dim::Int) = inds(a)[dim]

isnamed(::Type{<:AbstractITensor}) = true

function dim(a::AbstractITensor, n)
    return findfirst(==(name(n)), dimnames(a))
end
dims(a::AbstractITensor, ns) = Base.Fix1(dim, a).(ns)

dimname_isequal(x) = Base.Fix1(dimname_isequal, x)
dimname_isequal(x, y) = isequal(x, y)

dimname_isequal(r1::AbstractNamedArray, r2::AbstractNamedArray) = isequal(r1, r2)
dimname_isequal(r1::AbstractNamedArray, r2) = name(r1) == r2
dimname_isequal(r1, r2::AbstractNamedArray) = r1 == name(r2)

dimname_isequal(r1::AbstractNamedArray, r2::Name) = name(r1) == name(r2)
dimname_isequal(r1::Name, r2::AbstractNamedArray) = name(r1) == name(r2)

dimname_isequal(r1::AbstractNamedUnitRange, r2::AbstractNamedUnitRange) = isequal(r1, r2)
dimname_isequal(r1::AbstractNamedUnitRange, r2) = name(r1) == r2
dimname_isequal(r1, r2::AbstractNamedUnitRange) = r1 == name(r2)

dimname_isequal(r1::AbstractNamedUnitRange, r2::Name) = name(r1) == name(r2)
dimname_isequal(r1::Name, r2::AbstractNamedUnitRange) = name(r1) == name(r2)

function to_inds(a::AbstractITensor, dims)
    is = Base.Fix1(dim, a).(name.(dims))
    return Base.Fix1(inds, a).(is)
end

# Generic construction of named dims arrays.

"""
    nameddims(a::AbstractArray, inds)

Construct a named dimensions array from an denamed array `a` and named dimensions `inds`.
"""
function nameddims(a::AbstractArray, inds)
    return nameddimsconstructor(a, inds)(a, inds)
end

#=
    nameddimsof(a::AbstractITensor, b::AbstractArray)

Construct a named dimensions array with the dimension names of `a`
and based on the type of `a` but with the data from `b`.
=#
function nameddimsof(a::AbstractITensor, b::AbstractArray)
    return nameddimsconstructorof(a)(b, dimnames(a))
end

# Interface inspired by
# [ConstructionBase.constructorof](https://github.com/JuliaObjects/ConstructionBase.jl).
nameddimsconstructorof(a::AbstractITensor) = nameddimsconstructorof(typeof(a))
function nameddimsconstructorof(type::Type{<:AbstractITensor})
    return unspecify_type_parameters(type)
end

# Output a constructor for a named dims array (that should accept and denamed array and
# a set of named dimensions/axes/indices) based on the dimension names.
function nameddimsconstructor(a::AbstractArray, dims)
    dimnames = name.(dims)
    isempty(dimnames) && return ITensor
    return mapreduce(nameddimsconstructor, combine_nameddimsconstructors, dimnames)
end

nameddimsconstructor(nameddim) = nameddimsconstructor(typeof(nameddim))
# Can overload this to get custom named dims array wrapper
# depending on the dimension name types, for example
# output an `ITensor` if the dimension names are `IndexName`s.
nameddimsconstructor(nameddimtype::Type) = ITensor
function nameddimsconstructor(nameddimtype::Type{<:AbstractNamedUnitRange})
    return nameddimsconstructor(nametype(nameddimtype))
end
function combine_nameddimsconstructors(
        ::Type{<:AbstractITensor}, ::Type{<:AbstractITensor}
    )
    return ITensor
end
combine_nameddimsconstructors(::Type{T}, ::Type{T}) where {T <: AbstractITensor} = T

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

Base.copy(a::AbstractITensor) = nameddimsof(a, copy(denamed(a)))
Base.zero(a::AbstractITensor) = nameddimsof(a, zero(denamed(a)))

# `CartesianIndices` of a named tensor is the parent's, via the named axes (as the
# `AbstractArray` fallback did through `axes`).
Base.CartesianIndices(a::AbstractITensor) = CartesianIndices(axes(a))

# Forward `conj` to the underlying so that graded axes flip their sector arrows.
# The default `AbstractArray` fallback would broadcast `conj` over elements without
# touching the axes, which silently changes the contraction convention for tensors
# with graded (dual-tagged) axes.
Base.conj(a::AbstractITensor) = nameddimsof(a, conj(denamed(a)))

# `LinearAlgebra.normalize` infers result eltype via `typeof(first(a)/nrm)`, which
# scalar-indexes block-structured storage. `a / norm(a, p)` already preserves names.
function LinearAlgebra.normalize(a::AbstractITensor, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end
function LinearAlgebra.normalize!(a::AbstractITensor, p::Real = 2)
    LinearAlgebra.normalize!(denamed(a), p)
    return a
end

# Elementwise and scalar arithmetic. `AbstractArray` routes these through
# broadcasting; supply them directly now that the supertype is gone.
Base.:+(a1::AbstractITensor, a2::AbstractITensor) = a1 .+ a2
Base.:-(a1::AbstractITensor, a2::AbstractITensor) = a1 .- a2
Base.:-(a::AbstractITensor) = broadcast(-, a)
Base.:*(a::AbstractITensor, x::Number) = a .* x
Base.:*(x::Number, a::AbstractITensor) = x .* a
Base.:/(a::AbstractITensor, x::Number) = a ./ x

# Forward `Random.randn!` / `Random.rand!` to the concrete storage so they
# see the runtime eltype.
function Random.randn!(rng::Random.AbstractRNG, a::AbstractITensor)
    Random.randn!(rng, denamed(a))
    return a
end
function Random.rand!(rng::Random.AbstractRNG, a::AbstractITensor)
    Random.rand!(rng, denamed(a))
    return a
end

function Base.copyto!(a_dest::AbstractITensor, a_src::AbstractITensor)
    a′_dest = denamed(a_dest)
    # TODO: Use `denamed` to do the permutations lazily.
    a′_src = dename(a_src, inds(a_dest))
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
Base.Array(a::AbstractITensor) = Array(denamed(a))
Base.Array{T}(a::AbstractITensor) where {T} = Array{T}(denamed(a))
Base.Array{T, N}(a::AbstractITensor) where {T, N} = Array{T, N}(denamed(a))
Base.AbstractArray{T}(a::AbstractITensor) where {T} = AbstractArray{T, ndims(a)}(a)
function Base.AbstractArray{T, N}(a::AbstractITensor) where {T, N}
    dest = similar(a, T)
    copyto_axcheck!(denamed(dest), denamed(a))
    return dest
end

function Base.axes(a::AbstractITensor)
    return inds(a)
end
function Base.size(a::AbstractITensor)
    return length.(axes(a))
end

function Base.length(a::AbstractITensor)
    return prod(size(a); init = 1)
end

# Circumvent issue when ndims isn't known at compile time.
Base.axes(a::AbstractITensor, d) = axes(a)[d]

# Circumvent issue when ndims isn't known at compile time.
Base.size(a::AbstractITensor, d) = size(a)[d]

# Circumvent issue when ndims isn't known at compile time.
Base.ndims(a::AbstractITensor) = ndims(denamed(a))

# Circumvent issue when eltype isn't known at compile time.
Base.eltype(a::AbstractITensor) = eltype(denamed(a))

using VectorInterface: VectorInterface, scalartype
# Circumvent issue when eltype isn't known at compile time.
VectorInterface.scalartype(a::AbstractITensor) = scalartype(denamed(a))

Base.axes(a::AbstractITensor, dimname::Name) = axes(a, dim(a, dimname))
Base.size(a::AbstractITensor, dimname::Name) = size(a, dim(a, dimname))

function similar_nameddims(a::AbstractITensor, elt::Type, ax)
    return nameddimsconstructorof(a)(
        similar(denamed(a), elt, denamed.(Tuple(ax))),
        name.(ax)
    )
end
function similar_nameddims(a::AbstractArray, elt::Type, ax)
    return nameddims(similar(a, elt, denamed.(Tuple(ax))), name.(ax))
end

# Base.similar gets the eltype at compile time.
Base.similar(a::AbstractITensor) = similar(a, eltype(a))
function Base.similar(a::AbstractITensor, elt::Type)
    return similar_nameddims(a, elt)
end
function similar_nameddims(a::AbstractITensor, elt::Type)
    return nameddimsof(a, similar(denamed(a), elt))
end

# This is defined explicitly since the Base version expects the eltype
# to be known at compile time, which isn't true for ITensors.
function Base.similar(
        a::AbstractArray, inds::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return similar(a, eltype(a), inds)
end

function Base.similar(
        a::AbstractArray, elt::Type,
        inds::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return similar_nameddims(a, elt, inds)
end

# Same entry points with a named-tensor prototype. An `AbstractITensor` is no longer
# an `AbstractArray`, so the methods above (which build a named tensor from a plain
# array prototype) no longer cover it.
function Base.similar(
        a::AbstractITensor,
        inds::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return similar(a, eltype(a), inds)
end
function Base.similar(
        a::AbstractITensor, elt::Type,
        inds::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return similar_nameddims(a, elt, inds)
end
function setinds(a::AbstractITensor, inds)
    return nameddimsconstructorof(a)(denamed(a), inds)
end

function replacedimnames(a::AbstractITensor, replacements::Pair...)
    new_dimnames = replace(dimnames(a), replacements...)
    return nameddims(denamed(a), new_dimnames)
end
function replacedimnames(f, a::AbstractITensor)
    new_dimnames = replace(f, dimnames(a))
    return nameddims(denamed(a), new_dimnames)
end
mapdimnames(f, a::AbstractITensor) = replacedimnames(f, a)

function replaceinds(a::AbstractITensor, replacements::Pair...)
    new_inds = replace(inds(a), replacements...)
    return denamed(a)[new_inds...]
end
function replaceinds(f, a::AbstractITensor)
    new_inds = replace(f, inds(a))
    return denamed(a)[new_inds...]
end
mapinds(f, a::AbstractITensor) = replaceinds(f, a)

# `Base.isempty(a::AbstractArray)` is defined as `length(a) == 0`,
# which involves comparing a named integer to an denamed integer
# which isn't well defined.
Base.isempty(a::AbstractITensor) = isempty(denamed(a))

# Define this on objects rather than types in case the wrapper type
# isn't known at compile time, like for the ITensor type.
Base.IndexStyle(a::AbstractITensor) = IndexStyle(denamed(a))
Base.eachindex(a::AbstractITensor) = eachindex(denamed(a))

# Iteration, keys, and pairs forward to the parent (these were previously inherited
# from `AbstractArray`).
Base.iterate(a::AbstractITensor, state...) = iterate(denamed(a), state...)
Base.keys(a::AbstractITensor) = keys(denamed(a))
Base.pairs(a::AbstractITensor) = pairs(denamed(a))

# Multi-argument `eachindex` dispatches on the named index style, as the
# `AbstractArray` version did.
function Base.eachindex(a1::AbstractITensor, a_rest::AbstractITensor...)
    return eachindex(IndexStyle(a1, a_rest...), a1, a_rest...)
end

# Cartesian indices with names attached.
struct NamedIndexCartesian <: IndexStyle end

# When multiple named dims arrays are involved, use the named
# dimensions.
function Base.IndexStyle(a1::AbstractITensor, a2::AbstractITensor)
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
        Indices <: Tuple{Vararg{AbstractNamedUnitRange, N}},
        Index <: Tuple{Vararg{NamedInteger, N}},
    } <: AbstractITensor{DimName}
    indices::Indices
    function NamedDimsCartesianIndices(indices::Tuple{Vararg{AbstractNamedUnitRange}})
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
Base.axes(I::NamedDimsCartesianIndices) = (only ∘ axes ∘ denamed).(I.indices)
Base.size(I::NamedDimsCartesianIndices) = (length ∘ denamed).(I.indices)

function Base.getindex(a::NamedDimsCartesianIndices{N}, I::Vararg{Int, N}) where {N}
    index = map(a.indices, I) do r, i
        return r[i]
    end
    return NamedDimsCartesianIndex(index)
end

function denamed(I::NamedDimsCartesianIndices)
    return CartesianIndices(denamed.(I.indices))
end

# Iterating yields `NamedDimsCartesianIndex`es. The generic `AbstractITensor`
# iteration forwards to `denamed`, which here is a plain `CartesianIndices`, so
# convert each parent index back through `getindex`.
function Base.iterate(I::NamedDimsCartesianIndices, state...)
    y = iterate(denamed(I), state...)
    isnothing(y) && return nothing
    cartesian, next_state = y
    return I[Tuple(cartesian)...], next_state
end

function Base.eachindex(
        ::NamedIndexCartesian,
        a1::AbstractITensor,
        a_rest::AbstractITensor...
    )
    all(a -> issetequal(inds(a1), inds(a)), a_rest) ||
        throw(NameMismatch("Dimension name mismatch $(inds.((a1, a_rest...)))."))
    # TODO: Check the shapes match.
    return NamedDimsCartesianIndices(inds(a1))
end

# Base version ignores dimension names.
# TODO: Use `mapreduce(isequal, &&, a1, a2)`?
function Base.isequal(a1::AbstractITensor, a2::AbstractITensor)
    issetequal(inds(a1), inds(a2)) || return false
    return isequal(denamed(a1), denamed(a2, inds(a1)))
end

# Base version ignores dimension names.
# TODO: Use `mapreduce(==, &&, a1, a2)`?
# TODO: Handle `missing` values properly.
function Base.:(==)(a1::AbstractITensor, a2::AbstractITensor)
    issetequal(inds(a1), inds(a2)) || return false
    return denamed(a1) == denamed(a2, inds(a1))
end

# Base version ignores dimension names.
function Base.isapprox(a1::AbstractITensor, a2::AbstractITensor; kwargs...)
    issetequal(inds(a1), inds(a2)) || return false
    return isapprox(denamed(a1), denamed(a2, inds(a1)); kwargs...)
end

# Generalization of `Base.sort` to Tuples for Julia v1.10 compatibility.
# TODO: Remove when we drop support for Julia v1.10.
_sort(x; kwargs...) = sort(x; kwargs...)
_sort(x::NTuple{N}; kwargs...) where {N} = NTuple{N}(sort(collect(x); kwargs...))

function Base.hash(a::AbstractITensor, h::UInt64)
    h = hash(:ITensor, h)
    a′ = aligneddims(a, _sort(dimnames(a)))
    h = hash(denamed(a′), h)
    for i in inds(a′)
        h = hash(i, h)
    end
    return h
end

# Indexing.

# Scalar indexing

Base.firstindex(a::AbstractITensor) = firstindex(denamed(a))
Base.lastindex(a::AbstractITensor) = lastindex(denamed(a))

function Base.firstindex(a::AbstractITensor, d)
    return FirstIndex(a, d)
end

function Base.lastindex(a::AbstractITensor, d)
    return LastIndex(a, d)
end

# Redefine generic definition which expects `axes(a)` to
# return a Tuple.
function Base.to_indices(a::AbstractITensor, I::Tuple)
    return to_indices(a, Tuple(axes(a)), I)
end
# Fix ambiguity error with Base.
function Base.to_indices(
        a::AbstractITensor,
        I::Tuple{Union{Integer, CartesianIndex}}
    )
    return to_indices(a, Tuple(axes(a)), I)
end
function Base.checkbounds(::Type{Bool}, a::AbstractITensor, I::Int...)
    return checkbounds(Bool, denamed(a), I...)
end

function Base.to_indices(
        a::AbstractITensor, I::Tuple{NamedInteger, Vararg{NamedInteger}}
    )
    perm = getperm(name.(I), dimnames(a))
    # TODO: Throw a `NameMismatch` error.
    @assert isperm(perm)
    I = map(p -> I[p], perm)
    return map(inds(a), I) do dimname, i
        return checked_indexin(denamed(i), denamed(dimname))
    end
end
function Base.to_indices(
        a::AbstractITensor, I::Tuple{Pair{<:Any, Int}, Vararg{Pair{<:Any, Int}}}
    )
    inds = to_inds(a, first.(I))
    return to_indices(a, map((i, name) -> name[i], last.(I), inds))
end
function Base.to_indices(a::AbstractITensor, I::Tuple{Pair, Vararg{Pair}})
    inds = to_inds(a, first.(I))
    return map((i, name) -> name[i], last.(I), inds)
    return to_indices(a, named.(last.(I), first.(I)))
end

function Base.to_indices(a::AbstractITensor, I::Tuple{NamedDimsCartesianIndex})
    return to_indices(a, Tuple(only(I)))
end

function Base.getindex(a::AbstractITensor, I...)
    return getindex(a, to_indices(a, I)...)
end

function Base.getindex(a::AbstractITensor, I1::Int, Irest::Int...)
    return getindex(denamed(a), I1, Irest...)
end
function Base.getindex(
        a::AbstractITensor, I1::NamedInteger, Irest::NamedInteger...
    )
    return getindex(a, to_indices(a, (I1, Irest...))...)
end
function Base.getindex(a::AbstractITensor)
    return getindex(denamed(a))
end
# Linear indexing.
function Base.getindex(a::AbstractITensor, I::Int)
    return getindex(denamed(a), I)
end

function Base.setindex!(a::AbstractITensor, value, I1::Int, Irest::Int...)
    setindex!(denamed(a), value, I1, Irest...)
    return a
end
function Base.setindex!(a::AbstractITensor, value, I::CartesianIndex)
    setindex!(a, value, to_indices(a, (I,))...)
    return a
end

function Base.setindex!(
        a::AbstractITensor, value, I1::NamedInteger,
        Irest::NamedInteger...
    )
    setindex!(a, value, to_indices(a, (I1, Irest...))...)
    return a
end
function Base.setindex!(a::AbstractITensor, value, I::NamedDimsCartesianIndex)
    setindex!(a, value, to_indices(a, (I,))...)
    return a
end
function Base.setindex!(a::AbstractITensor, value, I1::Pair, Irest::Pair...)
    setindex!(a, value, to_indices(a, (I1, Irest...))...)
    return a
end
function Base.setindex!(a::AbstractITensor, value)
    setindex!(denamed(a), value)
    return a
end
# Linear indexing.
function Base.setindex!(a::AbstractITensor, value, I::Int)
    setindex!(denamed(a), value, I)
    return a
end

function Base.isassigned(a::AbstractITensor, I::Int...)
    return isassigned(denamed(a), I...)
end

# Slicing

# Like `const ViewIndex = Union{Real,AbstractArray}`.
const NamedViewIndex =
    Union{NamedInteger, AbstractNamedUnitRange, AbstractNamedArray}

using ArrayLayouts: ArrayLayouts, MemoryLayout

abstract type AbstractITensorLayout <: MemoryLayout end
struct ITensorLayout{ParentLayout} <: AbstractITensorLayout end

function ArrayLayouts.MemoryLayout(arrtype::Type{<:AbstractITensor})
    return ITensorLayout{typeof(MemoryLayout(parenttype(arrtype)))}()
end

function ArrayLayouts.sub_materialize(::ITensorLayout, a, ax)
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
function Base.getindex(a::Array, I1::AbstractNamedUnitRange)
    return copy(view(a, I1))
end
function Base.view(a::AbstractArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
    I = (I1, Irest...)
    return nameddims(view(a, denamed.(I)...), name.(I))
end

# TODO: Should this be a view?
function Base.getindex(a::AbstractITensor, I1::Name, Irest::Name...)
    return copy(view(a, I1, Irest...))
end
function Base.view(a::AbstractITensor, I1::Name, Irest::Name...)
    issetequal(dimnames(a), name.((I1, Irest...))) ||
        throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(name.((I1, Irest...)))."
        )
    )
    return a
end

function Base.getindex(a::AbstractITensor, I1::Pair, Irest::Pair...)
    return getindex(a, to_indices(a, (I1, Irest...))...)
end
function Base.view(a::AbstractITensor, I1::Pair, Irest::Pair...)
    I = (I1, Irest...)
    inds = to_inds(a, first.(I))
    return view(a, map((i, name) -> name[i], last.(I), inds)...)
end

function Base.getindex(
        a::AbstractITensor, I1::NamedViewIndex, Irest::NamedViewIndex...
    )
    return copy(view(a, I1, Irest...))
end
function Base.view(a::AbstractITensor, I1::NamedViewIndex, Irest::NamedViewIndex...)
    I = (I1, Irest...)
    perm = getperm(name.(I), dimnames(a))
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(name.(I))."
        )
    )
    Ip = map(p -> denamed(I[p]), perm)
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
# a = ITensor(rand(3,4), (:x, :y))
# b = view(a, 1:2, 2)
function view_nameddims(a::AbstractITensor, I...)
    nonscalar_dims = filter(dim -> !isscalarindex(I[dim]), ntuple(identity, ndims(a)))
    nonscalar_dimnames = map(dim -> dimnames(a, dim), nonscalar_dims)
    return nameddimsconstructorof(a)(view(denamed(a), I...), nonscalar_dimnames)
end

function Base.view(a::AbstractITensor, I::ViewIndex...)
    return view_nameddims(a, I...)
end

function getindex_nameddims(a::AbstractITensor, I...)
    return copy(view(a, I...))
end

function Base.getindex(a::AbstractITensor, I::ViewIndex...)
    return getindex_nameddims(a, I...)
end

function Base.setindex!(
        a::AbstractITensor,
        value::AbstractITensor,
        I1::NamedViewIndex,
        Irest::NamedViewIndex...
    )
    view(a, I1, Irest...) .= value
    return a
end
function Base.setindex!(
        a::AbstractITensor,
        value::AbstractArray,
        I1::NamedViewIndex,
        Irest::NamedViewIndex...
    )
    I = (I1, Irest...)
    a[I...] = nameddimsconstructorof(a)(value, name.(I))
    return a
end
function Base.setindex!(
        a::AbstractITensor,
        value::AbstractITensor,
        I1::ViewIndex,
        Irest::ViewIndex...
    )
    view(a, I1, Irest...) .= value
    return a
end
function Base.setindex!(
        a::AbstractITensor, value::AbstractArray, I1::ViewIndex, Irest::ViewIndex...
    )
    setindex!(denamed(a), value, I1, Irest...)
    return a
end

# Permute/align dimensions

function aligndims(a::AbstractITensor, dims)
    new_dimnames = name.(dims)
    perm = Tuple(getperm(dimnames(a), new_dimnames))
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(new_dimnames)."
        )
    )
    return nameddimsconstructorof(a)(permutedims(denamed(a), perm), new_dimnames)
end

function aligneddims(a::AbstractITensor, dims)
    new_dimnames = name.(dims)
    perm = getperm(dimnames(a), new_dimnames)
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(new_dimnames)."
        )
    )
    return nameddimsconstructorof(a)(
        permuteddims(denamed(a), perm), new_dimnames
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
                ax::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
            )
            a = $f′(rng, elt, denamed.(ax))
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
    for dimtype in [:NamedInteger, :AbstractNamedUnitRange]
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
for f in [:zeros, :ones], dimtype in [:NamedInteger, :AbstractNamedUnitRange]
    @eval begin
        function Base.$f(
                elt::Type{<:Number}, ax::Tuple{$dimtype, Vararg{$dimtype}}
            )
            a = $f(elt, denamed.(ax))
            return a[Name.(name.(ax))...]
        end
        function Base.$f(elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype})
            return $f(elt, (dim1, dims...))
        end
        Base.$f(dims::Tuple{$dimtype, Vararg{$dimtype}}) = $f(default_eltype(), dims)
        Base.$f(dim1::$dimtype, dims::Vararg{$dimtype}) = $f((dim1, dims...))
    end
end
for dimtype in [:NamedInteger, :AbstractNamedUnitRange]
    @eval begin
        function Base.fill(value, ax::Tuple{$dimtype, Vararg{$dimtype}})
            a = fill(value, denamed.(ax))
            return a[Name.(name.(ax))...]
        end
        function Base.fill(value, dim1::$dimtype, dims::Vararg{$dimtype})
            return fill(value, (dim1, dims...))
        end
    end
end

function Base.fill!(a::AbstractITensor, v)
    fill!(denamed(a), v)
    return a
end

function Base.map!(f, a_dest::AbstractITensor, a_srcs::AbstractITensor...)
    a′_dest = denamed(a_dest)
    # TODO: Use `denamed` to do the permutations lazily.
    # TODO: Define `dename[d](dimnames) = Base.Fix1(dename[d], dimnames)` and use it here?
    a′_srcs = Base.Fix2(dename, dimnames(a_dest)).(a_srcs)
    map!(f, a′_dest, a′_srcs...)
    return a_dest
end

function Base.map(f, a_srcs::AbstractITensor...)
    # copy(mapped(f, a_srcs...))
    return f.(a_srcs...)
end

function Base.mapreduce(f, op, a::AbstractITensor; kwargs...)
    return mapreduce(f, op, denamed(a); kwargs...)
end

# `sum` is routed to the underlying data rather than left to fall back on the
# `mapreduce` method above because some array types (such as graded arrays) define
# `Base.sum` directly but not the general `mapreduce`, so the denamed `sum` is the
# path that works for them.
function Base.sum(a::AbstractITensor; kwargs...)
    return sum(denamed(a); kwargs...)
end

function LinearAlgebra.promote_leaf_eltypes(a::AbstractITensor)
    return LinearAlgebra.promote_leaf_eltypes(denamed(a))
end

# Printing

# Copy of `Base.dims2string` defined in `show.jl`.
function dims_to_string(d)
    isempty(d) && return "0-dimensional"
    length(d) == 1 && return "$(d[1])-element"
    return join(map(string, d), '×')
end

using TypeParameterAccessors: type_parameters, unspecify_type_parameters
function concretetype_to_string_truncated(
        type::Type;
        param_truncation_length = typemax(Int)
    )
    isconcretetype(type) || throw(ArgumentError("Type must be concrete."))
    alias = Base.make_typealias(type)
    base_type, params = if isnothing(alias)
        unspecify_type_parameters(type), type_parameters(type)
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

function Base.summary(io::IO, a::AbstractITensor)
    print(io, dims_to_string(inds(a)))
    print(io, ' ')
    print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length = 40))
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", a::AbstractITensor)
    summary(io, a)
    println(io, ":")
    show(io, mime, denamed(a))
    return nothing
end

function Base.show(io::IO, a::AbstractITensor)
    show(io, denamed(a))
    print(io, "[", join(inds(a), ", "), "]")
    return nothing
end
