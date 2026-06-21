module ITensorBase

export ITensor, Index, aligndims, dimnametype, named, nameddims,
    operator, similar_operator
using Compat: @compat
@compat public to_inds
@compat public @names

# Named-array machinery (relocated from NamedDimsArrays.jl).
include("isnamed.jl")
include("randname.jl")
include("name.jl")
include("named.jl")
include("abstractnamedarray.jl")
include("namedarray.jl")
include("namedunitrange.jl")
include("abstractitensor.jl")
include("broadcast.jl")
include("tensoralgebra.jl")
include("linearalgebra.jl")
include("itensor.jl")
include("itensoroperator.jl")

# `IndexName` dimname flavor and the `Index` named unit range.
include("index.jl")
include("quirks.jl")

end
