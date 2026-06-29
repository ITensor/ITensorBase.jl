using ..ITensorBase: AbstractNamedTensor, ITensorBase, NamedUnitRange, getperm, inds, name,
    named, nameddims, unname, unnamed
using Base.Broadcast: Broadcast as BC, Broadcasted, broadcast_shape, broadcasted,
    check_broadcast_shape, combine_axes
using TensorAlgebra: TensorAlgebra as TA

abstract type AbstractNamedTensorStyle{N} <: BC.AbstractArrayStyle{N} end

# Both `NamedTensorStyle` and `NamedTensorOperatorStyle` are dynamically-ranked
# (`ndims(::AbstractNamedTensor) === Any`), so the rank parameter `N` is `Any`. The
# `Val{N}` constructors below are required by `Base.Broadcast` for ranked styles;
# they preserve the style and ignore the inferred rank.
struct NamedTensorStyle{N} <: AbstractNamedTensorStyle{N} end
NamedTensorStyle(::Val{N}) where {N} = NamedTensorStyle{N}()
NamedTensorStyle{M}(::Val{N}) where {M, N} = NamedTensorStyle{N}()

function BC.BroadcastStyle(arraytype::Type{<:AbstractNamedTensor})
    return NamedTensorStyle{ndims(arraytype)}()
end

# An `AbstractNamedTensor` broadcasts as itself (previously inherited from
# `AbstractArray`); without this the default `broadcastable` wraps it in a `Ref`.
BC.broadcastable(a::AbstractNamedTensor) = a

function BC.combine_axes(
        a1::AbstractNamedTensor, a_rest::AbstractNamedTensor...
    )
    return broadcast_shape(axes(a1), combine_axes(a_rest...))
end
function BC.combine_axes(a1::AbstractNamedTensor, a2::AbstractNamedTensor)
    return broadcast_shape(axes(a1), axes(a2))
end
BC.combine_axes(a::AbstractNamedTensor) = axes(a)

# The named axes are a `Tuple` of `NamedUnitRange`s. Dispatch the
# name-aware shape combination on that tuple form (the elements are not
# `AbstractUnitRange`s, so Base's positional tuple-shape methods do not apply).
function BC.broadcast_shape(
        ax1::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
        ax2::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
        ax_rest::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}...
    )
    return broadcast_shape(broadcast_shape(ax1, ax2), ax_rest...)
end

function BC.broadcast_shape(
        ax1::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
        ax2::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return promote_shape(ax1, ax2)
end

# Handle scalar values.
function BC.broadcast_shape(
        ax1::Tuple{}, ax2::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return ax2
end
function BC.broadcast_shape(
        ax1::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}, ax2::Tuple{}
    )
    return ax1
end

function Base.promote_shape(
        ax1::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
        ax2::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return set_promote_shape(ax1, ax2)
end

function set_promote_shape(
        ax1::Tuple{NamedUnitRange, Vararg{NamedUnitRange, N}},
        ax2::Tuple{NamedUnitRange, Vararg{NamedUnitRange, N}}
    ) where {N}
    perm = getperm(ax2, ax1)
    ax2_aligned = map(i -> ax2[i], perm)
    ax_promoted = promote_shape(unnamed.(ax1), unnamed.(ax2_aligned))
    return named.(ax_promoted, name.(ax1))
end

# Handle operations like `randn() + randn(2, 2)[i, j]``.
# TODO: Decide if this should be a general definition for `AbstractNamedTensor`,
# or just for `AbstractNamedTensor`.
function set_promote_shape(
        ax1::Tuple{}, ax2::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return ax2
end

# Handle operations like `randn(2, 2)[i, j] + randn()`.
# TODO: Decide if this should be a general definition for `AbstractNamedTensor`,
# or just for `AbstractNamedTensor`.
function set_promote_shape(
        ax1::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}, ax2::Tuple{}
    )
    return ax1
end

function BC.check_broadcast_shape(
        ax1::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
        ax2::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return set_check_broadcast_shape(ax1, ax2)
end

function set_check_broadcast_shape(
        ax1::Tuple{Any, Vararg{Any, N}},
        ax2::Tuple{Any, Vararg{Any, N}}
    ) where {N}
    perm = getperm(ax2, ax1)
    ax2_aligned = map(i -> ax2[i], perm)
    check_broadcast_shape(unnamed.(ax1), unnamed.(ax2_aligned))
    return nothing
end
set_check_broadcast_shape(ax1::Tuple{}, ax2::Tuple{}) = nothing

broadcasted_unnamed(x::Number, inds) = x
broadcasted_unnamed(a::AbstractNamedTensor, inds) = unnamed(a, inds)
function broadcasted_unnamed(bc::Broadcasted, inds)
    return broadcasted(bc.f, Base.Fix2(broadcasted_unnamed, inds).(bc.args)...)
end

# A bare (unnamed) array operand, used as an allocation prototype so a broadcast
# result inherits the operands' backend (e.g. graded) rather than a lazy permuted
# wrapper's `similar` (which can drop the backend).
unnamed_prototype(bc::Broadcasted) = unnamed_prototype(bc.args...)
unnamed_prototype(arg::AbstractNamedTensor, args...) = unnamed(arg)
unnamed_prototype(arg::Broadcasted, args...) = unnamed_prototype(arg.args..., args...)
unnamed_prototype(arg, args...) = unnamed_prototype(args...)

function Base.similar(bc::Broadcasted{<:AbstractNamedTensorStyle}, elt::Type, ax)
    inds_a = name.(ax)
    bc_unnamed = broadcasted_unnamed(bc, inds_a)
    a_unnamed = similar(bc_unnamed, elt)
    return nameddims(a_unnamed, inds_a)
end

inds(bc::Broadcasted) = name.(axes(bc))
function Base.copy(bc::Broadcasted{<:AbstractNamedTensorStyle})
    # We could use:
    # ```julia
    # elt = combine_eltypes(bc.f, bc.args)
    # copyto!(similar(bc, elt), bc)
    # ```
    # but `combine_eltypes` is based on type inference, which might fail.
    # Calling broadcasted on the unnamed arrays reuses the code logic in
    # Base.Broadcast for handling cases where type inference fails by determining
    # the output element type at runtime with widening.
    inds_dest = inds(bc)
    bc_unnamed = broadcasted_unnamed(bc, inds_dest)
    lb = TA.tryflattenlinear(bc_unnamed)
    if isnothing(lb)
        # Not a linear combination: ordinary fused broadcast.
        dest_unnamed = copy(bc_unnamed)
    else
        # Linear: lower to bipermutedimsopadd!. Allocate from an operand so the
        # result keeps the backend, using the backend's result axes (not `lb`'s).
        dest_axes = unnamed.(Tuple(axes(bc)))
        dest_unnamed = similar(unnamed_prototype(bc), eltype(lb), dest_axes)
        copyto!(dest_unnamed, lb)
    end
    return nameddims(dest_unnamed, inds_dest)
end

function Base.copyto!(
        dest::AbstractNamedTensor,
        bc::Broadcasted{<:AbstractNamedTensorStyle}
    )
    dest_unnamed = unnamed(dest)
    inds_dest = axes(dest)
    bc_unnamed = broadcasted_unnamed(bc, inds_dest)
    lb = TA.tryflattenlinear(bc_unnamed)
    if isnothing(lb)
        # Not a linear combination: ordinary fused broadcast.
        copyto!(dest_unnamed, bc_unnamed)
    else
        # Linear: lower to bipermutedimsopadd! into the existing dest.
        copyto!(dest_unnamed, lb)
    end
    return dest
end

# Operator-preserving broadcasting.
#
# An `NamedTensorOperator` broadcasts as itself (it does not peel to its `state`), so
# `op .+ op`, `2 .* op`, etc. carry the `NamedTensorOperatorStyle`. The style-combination
# rules below enforce the input rules declaratively:
#   - operator ⊗ operator → operator (preserved),
#   - operator ⊗ scalar → operator (`2 .* op` stays an operator),
#   - operator ⊗ non-operator tensor → error.
# The `BroadcastStyle(::Type{<:NamedTensorOperator})` mapping and the operator-specific
# `copy` / `similar` (which unwrap, delegate to `NamedTensorStyle`, then rewrap) live in
# `itensoroperator.jl`, where `NamedTensorOperator` is defined. `*` (contraction) is
# unchanged and still decays to `state`.

struct NamedTensorOperatorStyle{N} <: AbstractNamedTensorStyle{N} end
NamedTensorOperatorStyle(::Val{N}) where {N} = NamedTensorOperatorStyle{N}()
NamedTensorOperatorStyle{M}(::Val{N}) where {M, N} = NamedTensorOperatorStyle{N}()

# operator ⊗ operator stays an operator.
function BC.BroadcastStyle(
        ::NamedTensorOperatorStyle{M},
        ::NamedTensorOperatorStyle{N}
    ) where {M, N}
    return NamedTensorOperatorStyle{M}()
end
# operator ⊗ scalar (`DefaultArrayStyle{0}`, e.g. `2 .* op`) stays an operator.
function BC.BroadcastStyle(
        style::NamedTensorOperatorStyle, ::BC.DefaultArrayStyle{0}
    )
    return style
end
# operator ⊗ non-operator named tensor is type-nonsense and is rejected.
function BC.BroadcastStyle(::NamedTensorOperatorStyle, ::NamedTensorStyle)
    return throw(
        ArgumentError(
            "Cannot broadcast an `NamedTensorOperator` together with a non-operator " *
                "tensor. Wrap the tensor as an operator first, or unwrap the " *
                "operator with `state`."
        )
    )
end

# Reinterpret an operator-style `Broadcasted` under `NamedTensorStyle`, the broadcast
# over the operators' states, so the shared `NamedTensorStyle` implementation runs (its
# `broadcasted_unnamed` already peels each operator operand to its `state` via
# `unnamed`).
function statebroadcasted(bc::Broadcasted{<:NamedTensorOperatorStyle})
    return Broadcasted{NamedTensorStyle{Any}}(bc.f, bc.args, bc.axes)
end
