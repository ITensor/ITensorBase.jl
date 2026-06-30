module ITensorBaseMooncakeExt

using ITensorBase: AbstractNamedTensor, NamedUnitRange, dimnames, dimnames_setdiff,
    from_contract_labels, inds, name, nameperm, to_contract_labels, to_inds, uniquename
using Mooncake: Mooncake, @zero_derivative, DefaultCtx

Mooncake.tangent_type(::Type{<:NamedUnitRange}) = Mooncake.NoTangent

@zero_derivative DefaultCtx Tuple{typeof(nameperm), Any, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(to_contract_labels), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(from_contract_labels), Any, Any, Any}
# `dimnames(::NamedTensor)` returns the stored names `Vector` directly, so its output
# aliases a field, where `@zero_derivative` is documented to be incorrect. Let
# Mooncake differentiate it through the underlying `getfield`, whose built-in rule
# preserves the aliasing (the names are non-differentiable, so the result is zero).
@zero_derivative DefaultCtx Tuple{typeof(dimnames), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(dimnames_setdiff), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(inds), Any}
@zero_derivative DefaultCtx Tuple{typeof(inds), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(name), Any}
@zero_derivative DefaultCtx Tuple{typeof(uniquename), Any}
@zero_derivative DefaultCtx Tuple{typeof(uniquename), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(to_inds), Any, Any}

using ITensorBase: AbstractNamedTensor, NamedTensor, unnamed
using Mooncake: Tangent
function Base.copyto!(dest::NamedTensor, src::Tangent)
    # TODO: Account for the `inds` of the Tangent? In other words, is the tangent data
    # aligned with the `dest` data?
    copyto!(unnamed(dest), src.fields.parent)
    return dest
end

end
