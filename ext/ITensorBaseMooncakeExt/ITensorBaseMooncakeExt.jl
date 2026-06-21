module ITensorBaseMooncakeExt

using ITensorBase: AbstractITensor, NamedUnitRange, blockedperm_nameddims,
    combine_nameddimsconstructors, dimnames, dimnames_setdiff, inds, name,
    nameddimsconstructorof, randname, to_inds
using Mooncake: Mooncake, @zero_derivative, DefaultCtx
using TensorAlgebra: blockedperm

Mooncake.tangent_type(::Type{<:NamedUnitRange}) = Mooncake.NoTangent

@zero_derivative DefaultCtx Tuple{typeof(blockedperm), AbstractITensor, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(blockedperm_nameddims), Any, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(combine_nameddimsconstructors), Any, Any}
# `dimnames(::ITensor)` returns the stored names `Vector` directly, so its output
# aliases a field, where `@zero_derivative` is documented to be incorrect. Let
# Mooncake differentiate it through the underlying `getfield`, whose built-in rule
# preserves the aliasing (the names are non-differentiable, so the result is zero).
@zero_derivative DefaultCtx Tuple{typeof(dimnames), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(dimnames_setdiff), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(inds), Any}
@zero_derivative DefaultCtx Tuple{typeof(inds), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(name), Any}
@zero_derivative DefaultCtx Tuple{typeof(nameddimsconstructorof), Any}
@zero_derivative DefaultCtx Tuple{typeof(randname), Any}
@zero_derivative DefaultCtx Tuple{typeof(randname), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(to_inds), Any, Any}

using ITensorBase: AbstractITensor, ITensor, denamed
using Mooncake: Tangent
function Base.copyto!(dest::ITensor, src::Tangent)
    # TODO: Account for the `inds` of the Tangent? In other words, is the tangent data
    # aligned with the `dest` data?
    copyto!(denamed(dest), src.fields.parent)
    return dest
end

end
