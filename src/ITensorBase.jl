module ITensorBase

export ITensor, Index, NamedDimsArray, aligndims, dimnametype, named, nameddims,
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

# ITensor layer built on the named-array machinery.
include("index.jl")
include("abstractitensor.jl")
include("quirks.jl")

end
