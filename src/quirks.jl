using NamedDimsArrays: dename

# TODO: Deprecate, just use `Int(length(i))` or
# `unname(length(i))` directly.
# Conversion to `Int` is used in case the output is named.
dim(i::Index) = Int(length(i))
# TODO: Deprecate.
# Conversion to `Int` is used in case the output is named.
# TODO: Deprecate, just use `Int(length(i))` or
# `unname(length(i))` directly.
dim(a::AbstractITensor) = Int(length(a))

# TODO: Delete this definition?
Base.adjoint(i::Index) = prime(i)

# TODO: Deprecate, just use `randname` directly.
sim(n::IndexName) = randname(n)
sim(i::Index) = setname(i, sim(name(i)))
sim(a::AbstractITensor) = mapinds(sim, a)

# TODO: Maybe deprecate these and use `mapinds` directly?
prime(a::AbstractITensor) = mapinds(prime, a)
noprime(a::AbstractITensor) = mapinds(noprime, a)

# TODO: Delete these and just use set operations on `inds` directly.
function uniqueinds(a1::AbstractITensor, a_rest::AbstractITensor...)
    return setdiff(inds(a1), inds.(a_rest)...)
end
function uniqueind(a1::AbstractITensor, a_rest::AbstractITensor...)
    return only(uniqueinds(a1, a_rest...))
end
function commoninds(a1::AbstractITensor, a_rest::AbstractITensor...)
    return intersect(inds(a1), inds.(a_rest)...)
end
function commonind(a1::AbstractITensor, a_rest::AbstractITensor...)
    return only(commoninds(a1, a_rest...))
end

# TODO: Replace with a more general functionality in
# `GradedArrays`, like `isgraded`.
hasqns(r::AbstractUnitRange) = false
hasqns(i::Index) = hasqns(dename(i))
hasqns(a::AbstractITensor) = all(hasqns, inds(a))

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
        kwargs...,
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
        translate_factorize_kwargs(; kwargs...)...,
    )
end
