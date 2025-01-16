# TODO: Define this properly.
dag(i::Index) = i
# TODO: Deprecate.
dim(i::Index) = dename(length(i))
# TODO: Define this properly.
hasqns(i::Index) = false
# TODO: Deprecate.
itensor(parent::AbstractArray, nameddimsindices) = ITensor(parent, nameddimsindices)
function itensor(parent::AbstractArray, i1::Index, i_rest::Index...)
  return ITensor(parent, (i1, i_rest...))
end

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

using LinearAlgebra: svd
# TODO: Define this in `MatrixAlgebra.jl`/`TensorAlgebra.jl`.
function factorize(a::AbstractITensor, args...; kwargs...)
  U, S, V = svd(a, args...; kwargs...)
  return U, S * V
end
