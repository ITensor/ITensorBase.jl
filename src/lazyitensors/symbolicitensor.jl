# Expression leaf with no array payload, so it defines no `denamed`/`getindex`.
# It mirrors a plain `ITensor`'s `denamed` + `dimnames` split, but without an
# array: `denamed_axes` stands in for `axes(denamed(a))` (the underlying ranges)
# and `dimnames` holds the names, with `inds` reconstructed from the two.
# The axes are stored as a field rather than a type parameter, so symbolic
# tensors of different rank share one concrete type and a flat `Mul` over them
# stays concretely typed.
struct SymbolicITensor{DimName, Name} <: AbstractITensor{DimName}
    name::Name
    denamed_axes::Tuple
    dimnames::Vector{DimName}
end
function SymbolicITensor(symname, inds)
    dnames = collect(name.(inds))
    DimName = isempty(inds) ? typeof(symname) : eltype(dnames)
    return SymbolicITensor{DimName, typeof(symname)}(symname, denamed.(inds), dnames)
end

symname(a::SymbolicITensor) = getfield(a, :name)

dimnames(a::SymbolicITensor) = getfield(a, :dimnames)
function inds(a::SymbolicITensor)
    return named.(getfield(a, :denamed_axes), Tuple(getfield(a, :dimnames)))
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
