using Random: AbstractRNG, RandomDevice

# Generate a new unique name, for example in matrix factorizations.
# The randomness defaults to `Random.RandomDevice()` (OS entropy) rather than
# `Random.default_rng()`, so minting a name neither perturbs nor is perturbed by
# the numerical RNG (`Random.seed!` does not make ids reproducible), mirroring
# `UUIDs.uuid4`. An explicit `rng` can still be passed for reproducible tests.
uniquename(type::Type; kwargs...) = uniquename(RandomDevice(), type; kwargs...)
uniquename(rng::AbstractRNG, type::Type) = rand(rng, type)

uniquename(name; kwargs...) = uniquename(RandomDevice(), name; kwargs...)
uniquename(rng::AbstractRNG, name; kwargs...) = uniquename(rng, typeof(name); kwargs...)

function uniquename(rng::AbstractRNG, ::Type{<:AbstractString})
    return string(rand(rng, UInt128); base = 62)
end
uniquename(rng::AbstractRNG, ::Type{Symbol}) = Symbol(uniquename(rng, String))
