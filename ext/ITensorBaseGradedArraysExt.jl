module ITensorBaseGradedArraysExt

using GradedArrays: SectorRange
using ITensorBase: ITensorBase, name, nameddims, uniquename, unnamed
using Random: AbstractRNG, default_rng
using TensorKitSectors: Sector

const NamedUnitRange = ITensorBase.NamedUnitRange

# Flux-canceling constructors at the `Index` level: delegate to the GradedArrays flux backend on
# the unnamed axes, then reattach names, so the flux convention lives only in the backend. The
# sector may be a bare `TensorKitSectors.Sector` or a `SectorRange`; this is an extension because
# ITensorBase does not depend on the sector types.

# Name the delegated result: the physical-leg names followed by a fresh name for the dangling aux
# leg, minted of the legs' name type (not hardcoded to `IndexName`).
function nameddims_aux(a, codomain, domain)
    dimnames = name.((codomain..., domain...))
    aux_name = uniquename(eltype(dimnames))
    return nameddims(a, (dimnames..., aux_name))
end

# Three signature groups, each carrying a named physical axis so overloading `Base` is not piracy:
# nonempty codomain with a (possibly empty) domain, the codomain-only form, and empty codomain with
# a nonempty domain. The all-empty flux-only case has no named leg and is left to the backend.
for S in (Sector, SectorRange)
    # Nonempty codomain, domain given (possibly empty).
    for f in (:rand, :randn)
        @eval begin
            function Base.$f(
                    rng::AbstractRNG, elt::Type{<:Number}, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}}
                )
                a = Base.$f(rng, elt, c, unnamed.(codomain), unnamed.(domain))
                return nameddims_aux(a, codomain, domain)
            end
            function Base.$f(
                    rng::AbstractRNG, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}}
                )
                return Base.$f(rng, ITensorBase.default_eltype(), c, codomain, domain)
            end
            function Base.$f(
                    elt::Type{<:Number}, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}}
                )
                return Base.$f(default_rng(), elt, c, codomain, domain)
            end
            function Base.$f(
                    c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}}
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
                    domain::Tuple{Vararg{NamedUnitRange}}
                )
                a = Base.$f(elt, c, unnamed.(codomain), unnamed.(domain))
                return nameddims_aux(a, codomain, domain)
            end
            function Base.$f(
                    c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                    domain::Tuple{Vararg{NamedUnitRange}}
                )
                return Base.$f(ITensorBase.default_eltype(), c, codomain, domain)
            end
        end
    end
    @eval function Base.fill(
            value, c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
            domain::Tuple{Vararg{NamedUnitRange}}
        )
        a = Base.fill(value, c, unnamed.(codomain), unnamed.(domain))
        return nameddims_aux(a, codomain, domain)
    end
    # Codomain-only: the domain-omitted form, equivalent to an empty domain.
    for f in (:rand, :randn)
        @eval begin
            function Base.$f(
                    rng::AbstractRNG, elt::Type{<:Number}, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                return Base.$f(rng, elt, c, codomain, ())
            end
            function Base.$f(
                    rng::AbstractRNG, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                return Base.$f(rng, c, codomain, ())
            end
            function Base.$f(
                    elt::Type{<:Number}, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                return Base.$f(elt, c, codomain, ())
            end
            function Base.$f(c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}})
                return Base.$f(c, codomain, ())
            end
        end
    end
    for f in (:zeros, :ones)
        @eval begin
            function Base.$f(
                    elt::Type{<:Number}, c::$S,
                    codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                return Base.$f(elt, c, codomain, ())
            end
            function Base.$f(c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}})
                return Base.$f(c, codomain, ())
            end
        end
    end
    @eval function Base.fill(
            value, c::$S, codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
        )
        return Base.fill(value, c, codomain, ())
    end
    # Empty codomain, nonempty domain.
    for f in (:rand, :randn)
        @eval begin
            function Base.$f(
                    rng::AbstractRNG, elt::Type{<:Number}, c::$S,
                    codomain::Tuple{}, domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                a = Base.$f(rng, elt, c, unnamed.(codomain), unnamed.(domain))
                return nameddims_aux(a, codomain, domain)
            end
            function Base.$f(
                    rng::AbstractRNG, c::$S,
                    codomain::Tuple{}, domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                return Base.$f(rng, ITensorBase.default_eltype(), c, codomain, domain)
            end
            function Base.$f(
                    elt::Type{<:Number}, c::$S,
                    codomain::Tuple{}, domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                return Base.$f(default_rng(), elt, c, codomain, domain)
            end
            function Base.$f(
                    c::$S, codomain::Tuple{},
                    domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
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
                    codomain::Tuple{}, domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                a = Base.$f(elt, c, unnamed.(codomain), unnamed.(domain))
                return nameddims_aux(a, codomain, domain)
            end
            function Base.$f(
                    c::$S, codomain::Tuple{},
                    domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
                )
                return Base.$f(ITensorBase.default_eltype(), c, codomain, domain)
            end
        end
    end
    @eval function Base.fill(
            value, c::$S, codomain::Tuple{},
            domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
        )
        a = Base.fill(value, c, unnamed.(codomain), unnamed.(domain))
        return nameddims_aux(a, codomain, domain)
    end
end

end
