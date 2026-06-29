using WrappedUnions: @wrapped

@wrapped struct LazyNamedTensor{
        DimName, A <: AbstractNamedTensor{DimName},
    } <: AbstractNamedTensor{DimName}
    union::Union{A, Mul{LazyNamedTensor{DimName, A}}}
end

parenttype(::Type{LazyNamedTensor{DimName, A}}) where {DimName, A} = A
function parenttype(::Type{LazyNamedTensor{DimName}}) where {DimName}
    return AbstractNamedTensor{DimName}
end
parenttype(::Type{LazyNamedTensor}) = AbstractNamedTensor

function LazyNamedTensor(a::AbstractNamedTensor)
    return LazyNamedTensor{dimnametype(typeof(a)), typeof(a)}(a)
end
function LazyNamedTensor(a::Mul{L}) where {L <: LazyNamedTensor}
    return LazyNamedTensor{dimnametype(L), parenttype(L)}(a)
end
lazy(a::LazyNamedTensor) = a
lazy(a::AbstractNamedTensor) = LazyNamedTensor(a)
lazy(a::Mul{<:LazyNamedTensor}) = LazyNamedTensor(a)

dimnames(a::LazyNamedTensor) = dimnames_lazy(a)
inds(a::LazyNamedTensor) = inds_lazy(a)
# `axes` is computed from `inds_lazy` rather than the generic `unnamed`-based fallback
# because a `Mul` expression has no materialized `unnamed` array to take axes of.
Base.axes(a::LazyNamedTensor) = Tuple(inds_lazy(a))
unnamed(a::LazyNamedTensor) = unnamed_lazy(a)

# Broadcasting
function Base.BroadcastStyle(::Type{<:LazyNamedTensor})
    return LazyNamedTensorStyle()
end

# Derived functionality.
function TermInterface.maketerm(type::Type{LazyNamedTensor}, head, args, metadata)
    return maketerm_lazy(type, head, args, metadata)
end
Base.getindex(a::LazyNamedTensor, I::Int...) = getindex_lazy(a, I...)
TermInterface.arguments(a::LazyNamedTensor) = arguments_lazy(a)
TermInterface.children(a::LazyNamedTensor) = children_lazy(a)
TermInterface.head(a::LazyNamedTensor) = head_lazy(a)
TermInterface.iscall(a::LazyNamedTensor) = iscall_lazy(a)
TermInterface.isexpr(a::LazyNamedTensor) = isexpr_lazy(a)
TermInterface.operation(a::LazyNamedTensor) = operation_lazy(a)
TermInterface.sorted_arguments(a::LazyNamedTensor) = sorted_arguments_lazy(a)
AbstractTrees.children(a::LazyNamedTensor) = abstracttrees_children_lazy(a)
TermInterface.sorted_children(a::LazyNamedTensor) = sorted_children_lazy(a)
ismul(a::LazyNamedTensor) = ismul_lazy(a)
AbstractTrees.nodevalue(a::LazyNamedTensor) = nodevalue_lazy(a)
Base.Broadcast.materialize(a::LazyNamedTensor) = materialize_lazy(a)
Base.copy(a::LazyNamedTensor) = copy_lazy(a)
Base.:(==)(a1::LazyNamedTensor, a2::LazyNamedTensor) = equals_lazy(a1, a2)
Base.isequal(a1::LazyNamedTensor, a2::LazyNamedTensor) = isequal_lazy(a1, a2)
Base.hash(a::LazyNamedTensor, h::UInt64) = hash_lazy(a, h)
map_arguments(f, a::LazyNamedTensor) = map_arguments_lazy(f, a)
substitute(a::LazyNamedTensor, substitutions) = substitute_lazy(a, substitutions)
AbstractTrees.printnode(io::IO, a::LazyNamedTensor) = printnode_lazy(io, a)
printnode_nameddims(io::IO, a::LazyNamedTensor) = printnode_lazy(io, a)
Base.show(io::IO, a::LazyNamedTensor) = show_lazy(io, a)
Base.show(io::IO, mime::MIME"text/plain", a::LazyNamedTensor) = show_lazy(io, mime, a)
Base.:*(a::LazyNamedTensor) = mul_lazy(a)
Base.:*(a1::LazyNamedTensor, a2::LazyNamedTensor) = mul_lazy(a1, a2)
Base.:+(a1::LazyNamedTensor, a2::LazyNamedTensor) = add_lazy(a1, a2)
Base.:-(a1::LazyNamedTensor, a2::LazyNamedTensor) = sub_lazy(a1, a2)
Base.:*(a1::Number, a2::LazyNamedTensor) = mul_lazy(a1, a2)
Base.:*(a1::LazyNamedTensor, a2::Number) = mul_lazy(a1, a2)
Base.:/(a1::LazyNamedTensor, a2::Number) = div_lazy(a1, a2)
Base.:-(a::LazyNamedTensor) = sub_lazy(a)

# `IndexName`-specialized alias, paralleling `ITensor = NamedTensor{IndexName}`.
const LazyITensor = LazyNamedTensor{IndexName}
