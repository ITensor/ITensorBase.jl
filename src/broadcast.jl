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

broadcasted_unnamed(x::Number, names) = x
function broadcasted_unnamed(a::AbstractNamedTensor, names)
    # An operand already aligned to the destination names (the first operand always, and the
    # common case for the rest) needs no permutation, avoiding a `getperm` allocation and the
    # identity `permuteddims` wrapper. Skipping it makes a small add several times slower.
    dimnames(a) == names && return unnamed(a)
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
function broadcasted_unnamed(bc::Broadcasted, names)
    return broadcasted(bc.f, Base.Fix2(broadcasted_unnamed, names).(bc.args)...)
end

# A bare (unnamed) array operand, used as an allocation prototype so a broadcast
# result inherits the operands' backend (e.g. graded) rather than a lazy permuted
# wrapper's `similar` (which can drop the backend).
unnamed_prototype(bc::Broadcasted) = unnamed_prototype(bc.args...)
unnamed_prototype(arg::AbstractNamedTensor, args...) = unnamed(arg)
unnamed_prototype(arg::Broadcasted, args...) = unnamed_prototype(arg.args..., args...)
unnamed_prototype(arg, args...) = unnamed_prototype(args...)

# Skip Base's shape-combination step: named broadcasts don't need the `NamedUnitRange` axis
# machinery. Name compatibility is handled by the per-operand alignment in `broadcasted_unnamed`
# (via `getperm`), and unnamed-shape compatibility by TensorAlgebra.
BC.instantiate(bc::Broadcasted{<:AbstractNamedTensorStyle}) = bc

# The destination dimension names of a broadcast are those of its first named operand.
# Sourcing them here (rather than from `axes(bc)`) keeps the named axes off the hot path.
_dimnames(a::AbstractNamedTensor, args...) = dimnames(a)
_dimnames(bc::Broadcasted, args...) = _dimnames(bc.args..., args...)
_dimnames(_, args...) = _dimnames(args...)
dimnames(bc::Broadcasted) = _dimnames(bc.args...)

# The result element type of a linear combination, from the concrete unnamed leaves at runtime.
# `eltype(::LinearBroadcasted)` uses `Base.promote_op`, which runs a live inference call here
# because the leaves wrap a named tensor's (non-inferrable) backing array, so promote the
# concrete `eltype`s instead.
_lineareltype(a::AbstractArray) = eltype(a)
function _lineareltype(s::TA.ScaledBroadcasted)
    return promote_type(typeof(TA.coeff(s)), _lineareltype(TA.unscaled(s)))
end
_lineareltype(s::TA.AddBroadcasted) = promote_type(map(_lineareltype, TA.addends(s))...)

function Base.copy(bc::Broadcasted{<:AbstractNamedTensorStyle})
    nms = dimnames(bc)
    dest_unnamed = _copy_unnamed(broadcasted_unnamed(bc, nms), unnamed_prototype(bc))
    return nameddims(dest_unnamed, nms)
end

# Function barrier: `broadcasted_unnamed` and `unnamed_prototype` produce concretely-typed
# values whose *inferred* types are abstract (the named backing array is abstract), so this
# call re-specializes on the concrete runtime types and everything below is type-stable
# (`eltype(lb)` is now inferrable, no runtime `promote_op`). Inlining the body into `copy`
# instead costs one extra allocation per call.
#
# Allocate from `axes(lb)`, the flattened expression's axes, rather than the prototype's own:
# an axis-changing operand (a `conj` leaf dualizes its axes) makes them differ, and the
# destination must match the expression. All axes go in the codomain (empty domain), the
# all-codomain output convention `@tensor` uses for an unbipartitioned left-hand side; on a
# non-bipartitioned backend (a dense array) `similar_map` with an empty domain is a plain
# `similar` over `axes(lb)`.
function _copy_unnamed(bc_unnamed, prototype)
    lb = TA.tryflattenlinear(bc_unnamed)
    isnothing(lb) && return copy(bc_unnamed)
    return copyto!(TA.similar_map(prototype, eltype(lb), axes(lb), ()), lb)
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
    _copyto_unnamed!(unnamed(dest), broadcasted_unnamed(bc, dimnames(dest)))
    return dest
end

# Function barrier mirroring `_copy_unnamed`: `unnamed(dest)` and `broadcasted_unnamed`
# have abstract inferred types (the named backing array is abstract), so this call
# re-specializes on the concrete runtime types and the flatten/lower below is type-stable.
function _copyto_unnamed!(dest_unnamed, bc_unnamed)
    lb = TA.tryflattenlinear(bc_unnamed)
    isnothing(lb) && return copyto!(dest_unnamed, bc_unnamed)
    return copyto!(dest_unnamed, lb)
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
