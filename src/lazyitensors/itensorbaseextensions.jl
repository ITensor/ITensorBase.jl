# Defined to avoid type piracy.
# TODO: Define a proper hash function
# in ITensorBase.jl, maybe one that is
# independent of the order of dimensions.
function _hash(a::ITensor, h::UInt64)
    h = hash(:ITensor, h)
    h = hash(unnamed(a), h)
    for i in inds(a)
        h = hash(i, h)
    end
    return h
end
function _hash(x, h::UInt64)
    return hash(x, h)
end

using AbstractTrees: AbstractTrees
# Only print the dimension names when printing with `AbstractTrees.print_tree`.
function AbstractTrees.printnode(io::IO, a::AbstractITensor)
    dimnames_a = "{" * join(map(s -> "\"$s\"", dimnames(a)), ", ") * "}"
    print(io, dimnames_a)
    return nothing
end

# Custom version of `AbstractTrees.printnode` to
# avoid type piracy when overloading on `AbstractITensor`.
# Method specializations (`LazyITensor`, `SymbolicITensor`) live in
# `lazyitensor.jl` and `symbolicitensor.jl`.
printnode_nameddims(io::IO, x) = AbstractTrees.printnode(io, x)
