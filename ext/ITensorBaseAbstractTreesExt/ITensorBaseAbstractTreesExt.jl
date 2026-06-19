module ITensorBaseAbstractTreesExt

using AbstractTrees: AbstractTrees
using ITensorBase: AbstractITensor, dimnames

# Only print the dimension names when printing with `AbstractTrees.print_tree`.
function AbstractTrees.printnode(io::IO, a::AbstractITensor)
    dimnames_a = "{" * join(map(s -> "\"$s\"", dimnames(a)), ", ") * "}"
    print(io, dimnames_a)
    return nothing
end

end
