module ITensorBaseTensorKitExt

using ITensorBase: ITensorBase, NamedTensor, NamedUnitRange
using Random: AbstractRNG
using TensorAlgebra: TensorAlgebra
using TensorKit: TensorKit, AbstractTensorMap, ElementarySpace, dim, one, ⊗

# ================================  Index over a native space  ==============================
# A native TensorKit space is stored directly as the axis value of a `NamedUnitRange`, so
# `Index(V)` (and `dual`/`conj` of one) round-trips through the named layer with the space
# intact. These terminal constructors short-circuit `to_range`; the element type is `Int`
# (the flat index positions), matching how the space presents to TensorAlgebra as an axis.
function ITensorBase.NamedUnitRange{Name}(unnamed::ElementarySpace, name) where {Name}
    return ITensorBase.NamedUnitRange{Name, Int, typeof(unnamed)}(unnamed, name)
end
function ITensorBase.NamedUnitRange(unnamed::ElementarySpace, name)
    return ITensorBase.NamedUnitRange{typeof(name), Int, typeof(unnamed)}(unnamed, name)
end

# `conj(index)` lowers to `named(conj(space), name)`, so `named` on a space must rebuild a
# `NamedUnitRange` rather than fall through to the generic (array-shaped) `Named`.
ITensorBase.named(r::ElementarySpace, name) = ITensorBase.NamedUnitRange(r, name)

# The flat length of a space-backed axis is its total (dense) dimension, so `size`/`length`
# of a `TensorMap`-backed ITensor report dense dimensions.
function Base.length(r::NamedUnitRange{<:Any, <:Any, <:ElementarySpace})
    return dim(ITensorBase.unnamed(r))
end

# ===============================  cold-start construction  =================================
# Build an all-codomain `TensorMap` (trivial domain) from a tuple of native spaces;
# TensorAlgebra regroups into the bipartition each operation needs. The 4-arg
# `(codomain, domain)` form is used deliberately: the single-space `randn(rng, T, space)`
# method in TensorKit 0.17 mis-references the `domain` function and throws.
for (f, f′) in ((:randn, :_randn), (:rand, :_rand))
    @eval function ITensorBase.$f′(
            rng::AbstractRNG, elt::Type,
            dims::Tuple{ElementarySpace, Vararg{ElementarySpace}}
        )
        codomain = ⊗(dims...)
        return TensorKit.$f(rng, elt, codomain, one(codomain))
    end
end
function ITensorBase._zeros(
        elt::Type, dims::Tuple{ElementarySpace, Vararg{ElementarySpace}}
    )
    codomain = ⊗(dims...)
    return TensorKit.zeros(elt, codomain, one(codomain))
end

# =================================  wrap under names  ======================================
# The `AbstractArray`-typed `a[Name...]` / `nameddims` paths skip a non-array `TensorMap`
# parent; provide the `AbstractTensorMap` equivalents. `getindex` is not piracy: `Name` is
# an ITensorBase-owned argument type.
function Base.getindex(
        a::AbstractTensorMap, I1::ITensorBase.Name, Irest::ITensorBase.Name...
    )
    return ITensorBase.nameddims(a, ITensorBase.name.((I1, Irest...)))
end
ITensorBase.nameddims(a::AbstractTensorMap, inds) = NamedTensor(a, inds)

# ==========================  linear-combination broadcast  =================================
# ITensorBase lowers named broadcasting onto the raw parents, so a `TensorMap` operand needs a
# `BroadcastStyle` to broadcast lazily (Base would otherwise try to `collect` it). A linear
# combination then flattens (via TensorAlgebra's `tryflattenlinear`) to a `LinearBroadcasted`
# that materializes through `add!`/`bipermutedimsopadd!` (provided for `AbstractTensorMap` by
# the TensorAlgebra TensorKit extension). Element-wise (nonlinear) broadcast is not a
# meaningful operation on a symmetric tensor, so it errors rather than dense-converting.
struct TensorMapStyle <: Base.Broadcast.BroadcastStyle end
Base.Broadcast.BroadcastStyle(::Type{<:AbstractTensorMap}) = TensorMapStyle()
Base.Broadcast.BroadcastStyle(s::TensorMapStyle, ::TensorMapStyle) = s
Base.Broadcast.BroadcastStyle(s::TensorMapStyle, ::Base.Broadcast.BroadcastStyle) = s
Base.Broadcast.broadcastable(a::AbstractTensorMap) = a

# The linear-combination destination: TensorAlgebra's `copyto!(::AbstractArray, ::Linear...)`
# does not match a non-array `TensorMap`, so route to the in-place `add!` the design reserves
# for a backend to provide (`copyto!` here is not piracy: `LinearBroadcasted` is TA-owned).
function Base.copyto!(dest::AbstractTensorMap, src::TensorAlgebra.LinearBroadcasted)
    return TensorAlgebra.add!(dest, src, true, false)
end

# Nonlinear / element-wise broadcast falls here (`tryflattenlinear` returned `nothing`).
function Base.copy(::Base.Broadcast.Broadcasted{TensorMapStyle})
    return error(
        "element-wise broadcast is not supported for a `TensorMap`-backed `ITensor`; \
        only linear combinations such as `a .+ b` and `2 .* a` are supported"
    )
end

end
