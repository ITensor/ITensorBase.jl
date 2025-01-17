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

using LinearAlgebra: qr, svd
# TODO: Define this in `MatrixAlgebra.jl`/`TensorAlgebra.jl`.
function factorize(a::AbstractITensor, args...; maxdim=nothing, cutoff=nothing, kwargs...)
  if isnothing(maxdim) && isnothing(cutoff)
    Q, R = qr(a, args...)
    return Q, R
  else
    error("Truncation in `factorize` not implemented yet.")
    U, S, V = svd(a, args...; kwargs...)
    return U, S * V
  end
end

# TODO: Used in `ITensorMPS.jl`, decide where or if to define it.
# Ideally this would just be a zero-dimensional `ITensor` wrapping
# a special type, like `Zeros{UnspecifiedZero()}()`.
struct OneITensor <: AbstractITensor end
Base.size(::OneITensor) = ()
Base.:*(::OneITensor, ::OneITensor) = OneITensor()
Base.:*(::OneITensor, a::ITensor) = a
Base.:*(a::ITensor, ::OneITensor) = a
