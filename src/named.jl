# A named value is a tagged scalar: it pairs an underlying value with a name. The
# named-integer case `NamedInteger` is used for index values and array
# sizes. `Named` is standalone (not `<: Integer` or `<: Number`): mixed
# named/unnamed arithmetic and operations like `i1 * i2` are not cleanly definable
# under the numeric contract, and inherited fallbacks risk silently dropping the
# name. So it supplies the integer-like surface it needs directly.
struct Named{Name, Denamed}
    value::Denamed
    name::Name
end

# A named integer is just the integer case of `Named`. The alias gives it a
# readable name for dispatch (`NamedInteger`, `NamedInteger{IndexName}`, ...)
# without being a separate type.
const NamedInteger{Name, Denamed <: Integer} = Named{Name, Denamed}

# Minimal interface.
denamed(i::Named) = i.value
name(i::Named) = i.name

# Shorthand. Attaching a name to a scalar produces a `Named`; arrays and unit
# ranges have their own more specific `named` methods.
named(value, name) = Named(value, name)

# Derived interface.
setname(i::Named, name) = named(denamed(i), name)
setvalue(i::Named, value) = named(value, name(i))

denamedtype(::Type{<:Named{<:Any, Denamed}}) where {Denamed} = Denamed
nametype(::Type{<:Named{Name}}) where {Name} = Name

# Traits.
isnamed(::Type{<:Named}) = true

function Base.:(==)(i1::Named, i2::Named)
    return name(i1) == name(i2) && denamed(i1) == denamed(i2)
end
# Hash under a literal tag plus the denamed value and name. The tag is shared by an
# entire equality class (one for the scalar `Named`, one for all `AbstractNamedArray`
# types), not per concrete type, so `a == b => hash(a) == hash(b)` holds the way it
# does in Base (`[1, 2, 3] == 1:3` and they hash equally).
function hash_named(typetag::Symbol, x, h::UInt)
    h = hash(typetag, h)
    h = hash(denamed(x), h)
    return hash(name(x), h)
end
Base.hash(i::Named, h::UInt) = hash_named(:Named, i, h)

function randname(rng::AbstractRNG, i::Named)
    return named(denamed(i), randname(name(i)))
end

function Base.string(i::Named; kwargs...)
    return "named($(string(denamed(i); kwargs...)), $(repr(name(i))))"
end
function Base.show(io::IO, i::Named)
    print(io, "named(", denamed(i), ", ", repr(name(i)), ")")
    return nothing
end

# Integer interface, for the named-integer case `NamedInteger`.
# TODO: Should `*` make a random name, or require defining a way to combine names?
function Base.:*(i1::NamedInteger, i2::NamedInteger)
    return named(denamed(i1) * denamed(i2), fusednames(name(i1), name(i2)))
end
Base.:-(i::NamedInteger) = setvalue(i, -denamed(i))

## TODO: Support this, we need to define `NamedFloat`, `NamedReal`, `NamedNumber`, etc.
## This is used in `LinearAlgebra.norm`, for now we just overload that directly.
## Here, named numbers are treated as unitful, so multiplying them
## with denamed numbers means the result inherits the name.
## function Base.:*(i1::NamedInteger, i2::Number)
##   return named(denamed(i1) * i2, name(i1))
## end

# For the sake of generic code, the name is ignored.
# Used in `AbstractArray` `Base.show`.
# TODO: See if we can delete this.
Base.:+(i1::Int, i2::NamedInteger) = i1 + denamed(i2)

Base.:*(i1::Int, i2::NamedInteger) = named(i1 * denamed(i2), name(i2))

Base.zero(i::NamedInteger) = setvalue(i, zero(denamed(i)))
Base.one(i::NamedInteger) = setvalue(i, one(denamed(i)))
Base.signbit(i::NamedInteger) = signbit(denamed(i))
Base.unsigned(i::NamedInteger) = setvalue(i, unsigned(denamed(i)))

# Used in bounds checking when indexing with named dimensions.
function Base.:<(i1::NamedInteger, i2::NamedInteger)
    name(i1) == name(i2) || throw(NameMismatch("Mismatched names $(name(i1)), $(name(i2))"))
    return denamed(i1) < denamed(i2)
end
