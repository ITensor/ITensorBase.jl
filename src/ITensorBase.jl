module ITensorBase

export AbstractNamedTensor, NamedTensor, AbstractITensor, ITensor, Index,
    NamedUnitRange, aligndims, aligneddims, apply, codomainnames, commonind, commoninds,
    dimnames, dimnametype, domainnames, hascommoninds, id, inds, mapinds, named, nameddims,
    noncommonind, noncommoninds, noprime, operator, prime, replaceinds, sim,
    similar_operator, state,
    trycommonind, trynoncommonind, tryuniqueind, uniqueind, uniqueinds, unioninds,
    uniquename
if VERSION >= v"1.11.0-DEV.469"
    eval(
        Meta.parse(
            "public @names, IndexName, name, nametype, replacedimnames, setname, space, unnamed, unnamedtype, decoration, emptytags, gettag, gettags, hastag, plev, settags, tags, unsettags"
        )
    )
end

# Named-array machinery (relocated from NamedDimsArrays.jl).
include("isnamed.jl")
include("uniquename.jl")
include("name.jl")
include("named.jl")
include("abstractnamedarray.jl")
include("namedarray.jl")
include("namedunitrange.jl")
include("abstractnamedtensor.jl")
include("broadcast.jl")
include("tensoralgebra.jl")
include("linearalgebra.jl")
include("namedtensor.jl")
include("namedtensoroperator.jl")

# `IndexName` dimname flavor and the `Index` named unit range.
include("sorteddict.jl")
include("index.jl")
include("quirks.jl")

# Lazy and symbolic ITensor expressions.
include("lazyitensors/baseextensions.jl")
include("lazyitensors/itensorbaseextensions.jl")
include("lazyitensors/applied.jl")
include("lazyitensors/lazyinterface.jl")
include("lazyitensors/lazybroadcast.jl")
include("lazyitensors/lazyitensor.jl")
include("lazyitensors/symbolicitensor.jl")
include("lazyitensors/evaluation_order.jl")

end
