# TODO: Define this properly.
dag(i::Index) = i
# TODO: Deprecate.
dim(i::Index) = dename(length(i))
# TODO: Define this properly.
hasqns(i::Index) = false
inds(a::ITensor) = nameddimsindices(a)
# TODO: Deprecate.
itensor(parent::AbstractArray, nameddimsindices) = ITensor(parent, nameddimsindices)
function itensor(parent::AbstractArray, i1::Index, i_rest::Index...)
  return ITensor(parent, (i1, i_rest...))
end

Base.Broadcast.extrude(a::AbstractITensor) = a
