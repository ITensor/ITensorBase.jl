using ..ITensorBase:
    AbstractNamedTensor, ITensorBase, dimnames, getperm, named, nameddims, unnamed
using Base.Broadcast: Broadcast as BC, Broadcasted, broadcasted
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

# Unname a flattened named `LinearBroadcasted` preserving the operand's codomain/domain split. Valid
# only for a single-operand expression (`2 .* a`, `conj.(a)`): the sole leaf defines the output names,
# so it unnames to its bare backing. A sum has no single split, so its addends are aligned instead.
# (Flattening distributes scaling/conjugation over `+`, so a `Scaled`/`Conj` node never wraps an `Add`.)
function unnamed_split(a::TA.ScaledBroadcasted, names)
    return TA.linearbroadcasted(*, TA.coeff(a), unnamed_split(TA.unscaled(a), names))
end
function unnamed_split(a::TA.ConjBroadcasted, names)
    return TA.linearbroadcasted(conj, unnamed_split(parent(a), names))
end
unnamed_split(a::TA.AddBroadcasted, names) = unnamed_aligned(a, names)
unnamed_split(a::AbstractNamedTensor, names) = unnamed(a)

# Unname aligning every leaf to `names` through the `PermutedDims` wrapper (all-codomain output). Used
# for a sum's addends and for every in-place `copyto!` (aligned to the destination).
function unnamed_aligned(a::TA.ScaledBroadcasted, names)
    return TA.linearbroadcasted(*, TA.coeff(a), unnamed_aligned(TA.unscaled(a), names))
end
function unnamed_aligned(a::TA.ConjBroadcasted, names)
    return TA.linearbroadcasted(conj, unnamed_aligned(parent(a), names))
end
function unnamed_aligned(a::TA.AddBroadcasted, names)
    return TA.linearbroadcasted(+, map(x -> unnamed_aligned(x, names), TA.addends(a))...)
end
function unnamed_aligned(a::AbstractNamedTensor, names)
    return _broadcast_permuteddims(unnamed(a), getperm(dimnames(a), names))
end
# Broadcasting-only alignment: unlike the public `unnamed(a, names)` (which returns a
# `Base.PermutedDimsArray`, a full array), this wraps in `TensorAlgebra.PermutedDims`, which stores
# the permutation in a field rather than a type parameter, so it builds cheaply and type-stably
# from the runtime permutation and is a broadcast leaf the linear-combination fold absorbs via
# `bipermutedimsopadd!`. `PermutedDims` has almost no array interface, so it stays confined to this
# hot path and is never handed back to users. Function barrier: `unnamed(a)` is abstractly typed,
# so dispatching on the concrete array makes the rank a compile-time constant for the inferrable
# `ntuple(…, Val(ndims))` permutation. The rank comes from `TensorAlgebra.ndims`, which also
# covers non-`AbstractArray` backends like a `TensorMap`.
@noinline function _broadcast_permuteddims(array, perm)
    return TA.PermutedDims(array, ntuple(i -> perm[i], Val(TA.ndims(array))))
end
# Skip Base's shape-combination step: named broadcasts don't need the `NamedUnitRange` axis
# machinery. Name compatibility is handled by the per-operand alignment in `unnamed_aligned`
# (via `getperm`), and unnamed-shape compatibility by TensorAlgebra.
BC.instantiate(bc::Broadcasted{<:AbstractNamedTensorStyle}) = bc

# The destination dimension names of a broadcast are those of its first named operand.
# Sourcing them here (rather than from `axes(bc)`) keeps the named axes off the hot path.
_dimnames(a::AbstractNamedTensor, args...) = dimnames(a)
_dimnames(bc::Broadcasted, args...) = _dimnames(bc.args..., args...)
_dimnames(_, args...) = _dimnames(args...)
dimnames(bc::Broadcasted) = _dimnames(bc.args...)

function Base.copy(bc::Broadcasted{<:AbstractNamedTensorStyle})
    nms = dimnames(bc)
    return nameddims(_copy_unnamed(bc, nms), nms)
end

# Function barrier: `bc`'s named leaves are abstractly typed, so re-dispatching on the concrete `bc`
# here keeps the flatten/unname/materialize below type-stable. `copy(lb)` allocates through the unnamed
# backend's own broadcast-style `similar`, so the result inherits the backend (dense, graded, ...);
# `unnamed_split` keeps a single scaled/conjugated operand's codomain/domain split.
@noinline function _copy_unnamed(bc, nms)
    return copy(unnamed_split(TA.flattenlinear(bc), nms))
end

# `Base.Broadcast.materialize!` otherwise reconstructs the broadcast over `axes(dest)` and
# re-runs `instantiate`, forcing the `NamedUnitRange` axis machinery this style's `instantiate`
# no-op exists to skip (`combine_axes`/`set_promote_shape`). Route straight to `copyto!`, which
# aligns by dimname instead.
function BC.materialize!(
        dest::AbstractNamedTensor,
        bc::Broadcasted{<:AbstractNamedTensorStyle}
    )
    copyto!(dest, bc)
    return dest
end

function Base.copyto!(
        dest::AbstractNamedTensor,
        bc::Broadcasted{<:AbstractNamedTensorStyle}
    )
    _copyto_unnamed!(unnamed(dest), bc, dimnames(dest))
    return dest
end

# Function barrier mirroring `_copy_unnamed`. In place, so every operand aligns to `dest`.
@noinline function _copyto_unnamed!(dest_unnamed, bc, nms)
    return copyto!(dest_unnamed, unnamed_aligned(TA.flattenlinear(bc), nms))
end

# Operator-preserving broadcasting.
#
# An `NamedTensorOperator` broadcasts as itself (it does not peel to its `state`), so
# `op .+ op`, `2 .* op`, etc. carry the `NamedTensorOperatorStyle`. The style-combination
# rules below enforce the input rules declaratively:
#   - operator ⊗ operator → operator (preserved),
#   - operator ⊗ scalar → operator (`2 .* op` stays an operator),
#   - operator ⊗ non-operator tensor → operator (the tensor is a trivial, empty-pairing
#     operator, so the result inherits the operator operand's pairing).
# The `BroadcastStyle(::Type{<:NamedTensorOperator})` mapping and the operator-specific
# `copy` (which unwraps, delegates to `NamedTensorStyle`, then rewraps) live in
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
# operator ⊗ non-operator named tensor stays an operator: a plain tensor is a trivial
# operator with no pairing, so `o - t` (etc.) combines the states elementwise and the
# result inherits `o`'s output/input split (the split logic lives in
# `broadcast_operator_output_input`).
function BC.BroadcastStyle(style::NamedTensorOperatorStyle, ::NamedTensorStyle)
    return style
end

# Reinterpret an operator-style `Broadcasted` under `NamedTensorStyle`, the broadcast
# over the operators' states, so the shared `NamedTensorStyle` implementation runs (its
# `unnamed_split`/`unnamed_aligned` peel each operator operand to its `state` via `unnamed`).
function statebroadcasted(bc::Broadcasted{<:NamedTensorOperatorStyle})
    return Broadcasted{NamedTensorStyle{Any}}(bc.f, bc.args, bc.axes)
end
