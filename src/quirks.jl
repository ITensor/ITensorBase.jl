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

# TODO: This is just a stand-in for truncated SVD
# that only makes use of `maxdim`, just to get some
# functionality running in `ITensorMPS.jl`.
# Define a proper truncated SVD in
# `MatrixAlgebra.jl`/`TensorAlgebra.jl`.
function svd_truncated(a::AbstractITensor, codomain_inds; maxdim)
  U, S, V = svd(a, codomain_inds)
  r = Base.OneTo(min(maxdim, minimum(Int.(size(S)))))
  u = commonind(U, S)
  v = commonind(V, S)
  us = uniqueinds(U, S)
  vs = uniqueinds(V, S)
  U′ = U[(us .=> :)..., u => r]
  S′ = S[u => r, v => r]
  V′ = V[v => r, (vs .=> :)...]
  return U′, S′, V′
end

using LinearAlgebra: qr, svd
# TODO: Define this in `MatrixAlgebra.jl`/`TensorAlgebra.jl`.
function factorize(
  a::AbstractITensor, codomain_inds; maxdim=nothing, cutoff=nothing, ortho="left", kwargs...
)
  # TODO: Perform this intersection in `TensorAlgebra.qr`/`TensorAlgebra.svd`?
  # See https://github.com/ITensor/NamedDimsArrays.jl/issues/22.
  codomain_inds′ = if ortho == "left"
    intersect(inds(a), codomain_inds)
  elseif ortho == "right"
    setdiff(inds(a), codomain_inds)
  else
    error("Bad `ortho` input.")
  end
  F1, F2 = if isnothing(maxdim) && isnothing(cutoff)
    qr(a, codomain_inds′)
  else
    U, S, V = svd_truncated(a, codomain_inds′; maxdim)
    U, S * V
  end
  if ortho == "right"
    F2, F1 = F1, F2
  end
  return F1, F2, (; truncerr=zero(Bool),)
end
