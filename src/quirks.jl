# This seems to be needed to get broadcasting working.
# TODO: Investigate this and see if we can get rid of it.
Base.Broadcast.extrude(a::AbstractITensor) = a

# See: https://github.com/JuliaLang/julia/blob/v1.11.4/base/namedtuple.jl#L269
# `filter(f, ::NamedTuple)` is available in Julia v1.11, delete once
# we drop support for Julia v1.10.
filter_namedtuple(f, xs::NamedTuple) = xs[filter(k -> f(xs[k]), keys(xs))]

function translate_factorize_kwargs(;
        # MatrixAlgebraKit.jl/TensorAlgebra.jl kwargs.
        orth = nothing,
        rtol = nothing,
        maxrank = nothing,
        # ITensors.jl kwargs.
        ortho = nothing,
        cutoff = nothing,
        maxdim = nothing,
        kwargs...
    )
    orth = Symbol(@something orth ortho :left)
    rtol = @something rtol cutoff Some(nothing)
    maxrank = @something maxrank maxdim Some(nothing)
    trunc = (; rtol, maxrank)
    # !isnothing(maxrank) && error("`maxrank` not supported yet.")
    return filter_namedtuple(!isnothing, (; orth, trunc, kwargs...))
end

using TensorAlgebra: TensorAlgebra, factorize
function TensorAlgebra.factorize(a::AbstractITensor, codomain_inds, domain_inds; kwargs...)
    return invoke(
        factorize,
        Tuple{AbstractNamedDimsArray, Any, Any},
        a,
        codomain_inds,
        domain_inds;
        translate_factorize_kwargs(; kwargs...)...
    )
end
