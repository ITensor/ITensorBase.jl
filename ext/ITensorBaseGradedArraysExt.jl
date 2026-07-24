module ITensorBaseGradedArraysExt

using GradedArrays: SectorRange
using ITensorBase: ITensorBase, Index
using Random: AbstractRNG, default_rng
using TensorKitSectors: Sector

const NamedUnitRange = ITensorBase.NamedUnitRange

# Flux-canceling constructors at the `Index` level: mint a fresh auxiliary `Index` carrying
# sector `c` (auto-generated unique name) and forward to the map-shaped split constructor,
# appending the aux to the (implicitly dualized) domain dangling last, so the physical axes
# fuse to `c`. Mirrors the GradedArrays backend method. The sector may be a bare
# `TensorKitSectors.Sector` or a `SectorRange`. This lives in an extension because ITensorBase
# does not depend on the sector types.
for S in (Sector, SectorRange)
    for f in (:rand, :randn)
        @eval begin
            function Base.$f(
                    rng::AbstractRNG, elt::Type{<:Number}, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}} = ()
                )
                return Base.$f(rng, elt, codomain, (domain..., Index([c => 1])))
            end
            function Base.$f(
                    rng::AbstractRNG, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}} = ()
                )
                return Base.$f(rng, ITensorBase.default_eltype(), c, codomain, domain)
            end
            function Base.$f(
                    elt::Type{<:Number}, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}} = ()
                )
                return Base.$f(default_rng(), elt, c, codomain, domain)
            end
            function Base.$f(
                    c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}} = ()
                )
                return Base.$f(
                    default_rng(),
                    ITensorBase.default_eltype(),
                    c,
                    codomain,
                    domain
                )
            end
        end
    end
    for f in (:zeros, :ones)
        @eval begin
            function Base.$f(
                    elt::Type{<:Number}, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}} = ()
                )
                return Base.$f(elt, codomain, (domain..., Index([c => 1])))
            end
            function Base.$f(
                    c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}} = ()
                )
                return Base.$f(ITensorBase.default_eltype(), c, codomain, domain)
            end
        end
    end
    @eval function Base.fill(
            value, c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
            domain::Tuple{Vararg{NamedUnitRange}} = ()
        )
        return Base.fill(value, codomain, (domain..., Index([c => 1])))
    end
end

end
