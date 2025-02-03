module ITensorBaseSparseArraysBaseExt

using ITensorBase: ITensor, Index
using NamedDimsArrays: dename
using SparseArraysBase: SparseArraysBase, oneelement

function SparseArraysBase.oneelement(
  value, index::NTuple{N,Int}, ax::NTuple{N,Index}
) where {N}
  return ITensor(oneelement(value, index, only.(axes.(dename.(ax)))), ax)
end

end
