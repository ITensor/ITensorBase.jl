using NamedDimsArrays: dename

# TODO: Deprecate, just use `Int(length(i))` or
# `unname(length(i))` directly.
# Conversion to `Int` is used in case the output is named.
dim(i::Index) = Int(length(i))
# TODO: Deprecate.
# Conversion to `Int` is used in case the output is named.
# TODO: Deprecate, just use `Int(length(i))` or
# `unname(length(i))` directly.
dim(a::AbstractITensor) = Int(length(a))

# TODO: Replace with a more general functionality in
# `GradedArrays`, like `isgraded`.
hasqns(r::AbstractUnitRange) = false
hasqns(i::Index) = hasqns(dename(i))
hasqns(a::AbstractITensor) = all(hasqns, inds(a))

# This seems to be needed to get broadcasting working.
# TODO: Investigate this and see if we can get rid of it.
Base.Broadcast.extrude(a::AbstractITensor) = a

function translate_factorize_kwargs(;
  # ITensors.jl kwargs.
  ortho=nothing,
  cutoff=nothing,
  maxdim=nothing,
  # MatrixAlgebraKit.jl/TensorAlgebra.jl kwargs.
  orth=nothing,
  trunc=nothing,
  kwargs...,
)
  @show ortho, cutoff, maxdim
  @show orth, trunc
  @show kwargs
  return error()
end

using TensorAlgebra: TensorAlgebra, factorize
function TensorAlgebra.factorize(a::AbstractITensor, codomain_inds, domain_inds; kwargs...)
  return invoke(
    factorize,
    Tuple{AbstractNamedDimsArray,Any,Any},
    a,
    codomain_inds,
    domain_inds;
    translate_factorize_kwargs(; kwargs...)...,
  )
end
