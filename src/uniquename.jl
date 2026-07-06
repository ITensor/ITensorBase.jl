using Random: AbstractRNG, RandomDevice

"""
    uniquename([rng,] name)
    uniquename([rng,] type::Type)

Mint a fresh, unique name. Given an existing `name`, produce a new name of the same flavor
that is distinct from any other, for example to label a freshly generated dimension in a
matrix factorization. Decoration carried by the seed is kept: `uniquename` on an `IndexName`
keeps its tags and prime level, minting only a fresh id. Pass the name `type` instead of an
instance to mint a bare name (for `IndexName`, no tags and prime level zero), for a fresh
dimension that should not inherit any seed's decoration. Randomness defaults to OS entropy
(`Random.RandomDevice`) so that minting a name neither perturbs nor is perturbed by the
numerical RNG. Pass an explicit `rng` for a reproducible name.

# Examples

```jldoctest
julia> i = Index(2);

julia> uniquename(i) != i
true
```
"""
function uniquename end

# The randomness defaults to `Random.RandomDevice()` (OS entropy) rather than
# `Random.default_rng()`, so minting a name neither perturbs nor is perturbed by
# the numerical RNG (`Random.seed!` does not make ids reproducible), mirroring
# `UUIDs.uuid4`. An explicit `rng` can still be passed for reproducible tests.
uniquename(type::Type; kwargs...) = uniquename(RandomDevice(), type; kwargs...)
uniquename(rng::AbstractRNG, type::Type) = rand(rng, type)

uniquename(name; kwargs...) = uniquename(RandomDevice(), name; kwargs...)
uniquename(rng::AbstractRNG, name; kwargs...) = uniquename(rng, typeof(name); kwargs...)

function uniquename(rng::AbstractRNG, ::Type{T}) where {T <: AbstractString}
    # Base 62 (`[0-9A-Za-z]`) is the largest radix `string` supports, giving the
    # most compact identifier-safe name: 22 characters for 128 random bits.
    return convert(T, string(rand(rng, UInt128); base = 62))
end
uniquename(rng::AbstractRNG, ::Type{Symbol}) = Symbol(uniquename(rng, String))
