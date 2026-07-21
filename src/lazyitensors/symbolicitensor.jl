# Expression leaf with no array payload, so it defines no `unnamed`/`getindex`.
# A symbolic tensor is a placeholder substituted with a real tensor before
# contraction, so it only needs what drives contraction-order selection: the
# `dimnames` and the index `size`s (the cost model uses lengths). Its `axes` are
# reconstructed as plain ranges of those sizes. Storing sizes and dimnames as
# fields rather than type parameters lets symbolic tensors of different rank
# share one concrete type so a flat `Mul` over them stays concretely typed.
struct SymbolicNamedTensor{DimName, Name} <: AbstractNamedTensor{DimName}
    name::Name
    size::Vector{Int}
    dimnames::Vector{DimName}
end
function SymbolicNamedTensor(symname, inds)
    dnames = collect(name.(inds))
    DimName = isempty(inds) ? typeof(symname) : eltype(dnames)
    sizes = Int[length(i) for i in inds]
    return SymbolicNamedTensor{DimName, typeof(symname)}(symname, sizes, dnames)
end

symname(a::SymbolicNamedTensor) = getfield(a, :name)

dimnames(a::SymbolicNamedTensor) = getfield(a, :dimnames)
function Base.axes(a::SymbolicNamedTensor)
    return named.(Tuple(Base.OneTo.(getfield(a, :size))), Tuple(getfield(a, :dimnames)))
end
Base.ndims(a::SymbolicNamedTensor) = length(getfield(a, :dimnames))

function Base.:(==)(a::SymbolicNamedTensor, b::SymbolicNamedTensor)
    return symname(a) == symname(b) && dimnames(a) == dimnames(b)
end
Base.isequal(a::SymbolicNamedTensor, b::SymbolicNamedTensor) = a == b
function Base.hash(a::SymbolicNamedTensor, h::UInt64)
    h = hash(:SymbolicNamedTensor, h)
    h = hash(symname(a), h)
    return hash(dimnames(a), h)
end

# Products build lazy expressions rather than contracting numerically.
Base.:*(a::SymbolicNamedTensor, b::SymbolicNamedTensor) = lazy(a) * lazy(b)
Base.:*(a::SymbolicNamedTensor, b::LazyNamedTensor) = lazy(a) * b
Base.:*(a::LazyNamedTensor, b::SymbolicNamedTensor) = a * lazy(b)

issymbolic(a) = a isa SymbolicNamedTensor
issymbolic(a::LazyNamedTensor) = !iscall(a) && issymbolic(unwrap(a))

function Base.show(io::IO, a::SymbolicNamedTensor)
    print(io, symname(a))
    if ndims(a) > 0
        print(io, "[", join(dimnames(a), ","), "]")
    end
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::SymbolicNamedTensor)
    summary(io, a)
    println(io, ":")
    show(io, a)
    return nothing
end

# `IndexName`-specialized alias, paralleling `ITensor = NamedTensor{IndexName}`.
const SymbolicITensor = SymbolicNamedTensor{IndexName}

using AbstractTrees: AbstractTrees
function AbstractTrees.printnode(io::IO, a::SymbolicNamedTensor)
    show(io, a)
    return nothing
end

function symnameddims(symname, dims)
    return lazy(SymbolicNamedTensor(symname, dims))
end
symnameddims(name) = symnameddims(name, ())

function printnode_nameddims(io::IO, a::SymbolicNamedTensor)
    AbstractTrees.printnode(io, a)
    return nothing
end
