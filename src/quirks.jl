# TODO: Define this properly.
dag(i::Index) = i
# TODO: Define this properly.
dag(a::ITensor) = a
# TODO: Deprecate.
# Conversion to `Int` is used in case the output is named.
dim(i::Index) = Int(length(i))
# TODO: Deprecate.
# Conversion to `Int` is used in case the output is named.
dim(a::AbstractITensor) = Int(length(a))
# TODO: Define this properly.
hasqns(i::Index) = false
# TODO: Define this properly.
hasqns(i::AbstractITensor) = false
# TODO: Deprecate, and/or decide on aliasing behavior of `ITensor`.
itensor(parent::AbstractArray, nameddimsindices) = ITensor(parent, nameddimsindices)
function itensor(parent::AbstractArray, i1::Index, i_rest::Index...)
  return ITensor(parent, (i1, i_rest...))
end
# TODO: Deprecate.
order(a::AbstractArray) = ndims(a)
# TODO: Deprecate.
using NamedDimsArrays: aligndims
permute(a::AbstractITensor, dimnames) = aligndims(a, dimnames)

# This seems to be needed to get broadcasting working.
# TODO: Investigate this and see if we can get rid of it.
Base.Broadcast.extrude(a::AbstractITensor) = a

# TODO: Generalize this.
# Maybe define it as `oneelement`, and base it on
# `FillArrays.OneElement` (https://juliaarrays.github.io/FillArrays.jl/stable/#FillArrays.OneElement).
function onehot(iv::Pair{<:Index,<:Int})
  a = ITensor(first(iv))
  a[last(iv)] = one(Bool)
  return a
end

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

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
# Ideally this would just be a zero-dimensional `ITensor` wrapping
# a special type, like `Zeros{UnspecifiedZero()}()`.
struct OneITensor <: AbstractITensor end
Base.size(::OneITensor) = ()
Base.:*(::OneITensor, ::OneITensor) = OneITensor()
Base.:*(::OneITensor, a::ITensor) = a
Base.:*(a::ITensor, ::OneITensor) = a
