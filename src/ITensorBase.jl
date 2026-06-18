module ITensorBase

export ITensor, Index, aligndims, dimnametype, named, nameddims,
    operator, similar_operator
using Compat: @compat
@compat public to_inds
@compat public @names

# Named-array machinery (relocated from NamedDimsArrays.jl).
include("littleset.jl")
include("isnamed.jl")
include("randname.jl")
include("abstractnamedinteger.jl")
include("namedinteger.jl")
include("abstractnamedarray.jl")
include("namedarray.jl")
include("abstractnamedunitrange.jl")
include("namedunitrange.jl")
include("abstractnameddimsarray.jl")
include("broadcast.jl")
include("tensoralgebra.jl")
include("linearalgebra.jl")
include("nameddimsarray.jl")
include("nameddimsoperator.jl")

# `IndexName` dimname flavor and the `Index` named unit range.
include("index.jl")
include("quirks.jl")

end
