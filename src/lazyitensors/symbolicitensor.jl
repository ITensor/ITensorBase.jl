# Expression leaf with no array payload, so it defines no `unnamed`/`getindex`.
# A symbolic tensor is a placeholder substituted with a real tensor before
# contraction, so it only needs what drives contraction-order selection: the
# `dimnames` and the index `size`s (the cost model uses lengths). Its `axes` are
# reconstructed as plain ranges of those sizes. Storing sizes and dimnames as
# fields rather than type parameters lets symbolic tensors of different rank
# share one concrete type so a flat `Mul` over them stays concretely typed.
struct SymbolicITensor{DimName, Name} <: AbstractITensor{DimName}
    name::Name
    size::Vector{Int}
    dimnames::Vector{DimName}
end
function SymbolicITensor(symname, inds)
    dnames = collect(name.(inds))
    DimName = isempty(inds) ? typeof(symname) : eltype(dnames)
    sizes = Int[length(i) for i in inds]
    return SymbolicITensor{DimName, typeof(symname)}(symname, sizes, dnames)
end

symname(a::SymbolicITensor) = getfield(a, :name)

dimnames(a::SymbolicITensor) = getfield(a, :dimnames)
function Base.axes(a::SymbolicITensor)
    return named.(Tuple(Base.OneTo.(getfield(a, :size))), Tuple(getfield(a, :dimnames)))
end
dimnametype(::Type{<:SymbolicITensor{DimName}}) where {DimName} = DimName
Base.ndims(a::SymbolicITensor) = length(getfield(a, :dimnames))

function Base.:(==)(a::SymbolicITensor, b::SymbolicITensor)
    return symname(a) == symname(b) && dimnames(a) == dimnames(b)
end
Base.isequal(a::SymbolicITensor, b::SymbolicITensor) = a == b
function Base.hash(a::SymbolicITensor, h::UInt64)
    h = hash(:SymbolicITensor, h)
    h = hash(symname(a), h)
    return hash(dimnames(a), h)
end

# Products build lazy expressions rather than contracting numerically.
Base.:*(a::SymbolicITensor, b::SymbolicITensor) = lazy(a) * lazy(b)
Base.:*(a::SymbolicITensor, b::LazyITensor) = lazy(a) * b
Base.:*(a::LazyITensor, b::SymbolicITensor) = a * lazy(b)

issymbolic(a) = a isa SymbolicITensor
issymbolic(a::LazyITensor) = !iscall(a) && issymbolic(unwrap(a))

function Base.show(io::IO, a::SymbolicITensor)
    print(io, symname(a))
    if ndims(a) > 0
        print(io, "[", join(dimnames(a), ","), "]")
    end
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::SymbolicITensor)
    summary(io, a)
    println(io, ":")
    show(io, a)
    return nothing
end

using AbstractTrees: AbstractTrees
function AbstractTrees.printnode(io::IO, a::SymbolicITensor)
    show(io, a)
    return nothing
end

function symnameddims(symname, dims)
    return lazy(SymbolicITensor(symname, dims))
end
symnameddims(name) = symnameddims(name, ())

function printnode_nameddims(io::IO, a::SymbolicITensor)
    AbstractTrees.printnode(io, a)
    return nothing
end
