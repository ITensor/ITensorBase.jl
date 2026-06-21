using Random: Random, AbstractRNG, randstring

# Generate a new random name, for example in matrix
# factorizations.
uniquename(type::Type; kwargs...) = uniquename(Random.default_rng(), type; kwargs...)
uniquename(rng::AbstractRNG, type::Type) = rand(rng, type)

uniquename(name; kwargs...) = uniquename(Random.default_rng(), name; kwargs...)
uniquename(rng::AbstractRNG, name; kwargs...) = uniquename(rng, typeof(name); kwargs...)

uniquename(rng::AbstractRNG, ::Type{<:AbstractString}; length = 8) = randstring(rng, length)
function uniquename(rng::AbstractRNG, ::Type{Symbol}; kwargs...)
    return Symbol(uniquename(rng, String; kwargs...))
end
