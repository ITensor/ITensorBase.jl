module ITensorBaseOMEinsumContractionOrdersExt

using ITensorBase.TermInterface: arguments
using ITensorBase: ITensorBase, inds, ismul, optimize_contraction_order
using OMEinsumContractionOrders:
    OMEinsumContractionOrders, CodeOptimizer, EinCode, NestedEinsum, optimize_code

# Rebuild a nested product expression from an optimized `NestedEinsum`, mapping each
# leaf's `tensorindex` back to the corresponding argument of the flat product.
function nested_einsum_to_expr(f, code::NestedEinsum)
    # A leaf holds the 1-based index of its input tensor; internal nodes hold `-1`.
    return if code.tensorindex != -1
        f(code.tensorindex)
    else
        prod(Base.Fix1(nested_einsum_to_expr, f), code.args)
    end
end

# Find a contraction order with any OMEinsumContractionOrders optimizer (`GreedyMethod`,
# `TreeSA`, `KaHyParBipartite`, ...) by forwarding to `optimize_code`.
function ITensorBase.optimize_contraction_order(alg::CodeOptimizer, a)
    @assert ismul(a)
    ts = arguments(a)
    ixs = collect.(inds.(ts))
    all_inds = reduce(vcat, ixs)
    labels = unique(all_inds)
    size_dict = Dict(i => length(i) for i in labels)
    # Open indices (appearing on a single tensor) are the output of the network.
    iy = filter(i -> count(==(i), all_inds) == 1, labels)
    code = optimize_code(EinCode(ixs, iy), size_dict, alg)
    return nested_einsum_to_expr(i -> ts[i], code)
end

end
