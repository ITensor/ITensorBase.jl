module ITensorBaseDiagonalArraysExt

using DiagonalArrays: DiagonalArrays, δ, delta
using ITensorBase: ITensor, Index

# TODO: Define more generic definitions in `NamedDimsArraysDiagonalArraysExt`.
function DiagonalArrays.delta(elt::Type{<:Number}, is::Tuple{Index, Vararg{Index}})
    return ITensor(delta(elt, Int.(length.(is))), is)
end
function DiagonalArrays.δ(elt::Type{<:Number}, is::Tuple{Index, Vararg{Index}})
    return delta(elt, is)
end
DiagonalArrays.delta(is::Tuple{Index, Vararg{Index}}) = delta(Bool, is)
DiagonalArrays.δ(is::Tuple{Index, Vararg{Index}}) = delta(is)
function DiagonalArrays.delta(elt::Type{<:Number}, i1::Index, i_rest::Index...)
    return delta(elt, (i1, i_rest...))
end
function DiagonalArrays.δ(elt::Type{<:Number}, i1::Index, i_rest::Index...)
    return delta(elt, i1, i_rest...)
end
function DiagonalArrays.delta(i1::Index, i_rest::Index...)
    return delta((i1, i_rest...))
end
function DiagonalArrays.δ(i1::Index, i_rest::Index...)
    return delta(i1, i_rest...)
end

end
