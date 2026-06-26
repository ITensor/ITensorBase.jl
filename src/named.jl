# A named value is a tagged scalar: it pairs an underlying value with a name. The
# named-integer case `NamedInteger` is used for index values and array
# sizes. `Named` is standalone (not `<: Integer` or `<: Number`): mixed
# named/unnamed arithmetic and operations like `i1 * i2` are not cleanly definable
# under the numeric contract, and inherited fallbacks risk silently dropping the
# name. So it supplies the integer-like surface it needs directly.
struct Named{Name, Unnamed}
    value::Unnamed
    name::Name
end

# A named integer is just the integer case of `Named`. The alias gives it a
# readable name for dispatch (`NamedInteger`, `NamedInteger{IndexName}`, ...)
# without being a separate type.
const NamedInteger{Name, Unnamed <: Integer} = Named{Name, Unnamed}

"""
    name(a)

The name attached to a named object `a`, such as a `Named` scalar, a named array, or a
named unit range. This is the inverse of the name component of [`named`](@ref): `name`
recovers the name, [`unnamed`](@ref) recovers the value.

# Examples

```jldoctest
julia> using ITensorBase: name

julia> name(named(2, :i))
:i
```

See also [`named`](@ref), [`unnamed`](@ref), [`setname`](@ref).
"""
function name end

"""
    unnamed(a)

The underlying value of a named object `a`, with its name stripped off. This is the
inverse of the value component of [`named`](@ref): [`name`](@ref) recovers the name,
`unnamed` recovers the value. On an [`AbstractITensor`](@ref) it returns the underlying
unnamed array.

# Examples

```jldoctest
julia> using ITensorBase: unnamed

julia> unnamed(named(2, :i))
2
```

See also [`named`](@ref), [`name`](@ref).
"""
function unnamed end

"""
    setname(a, name)

Return a copy of the named object `a` with its name replaced by `name`, keeping the
underlying value unchanged.

# Examples

```jldoctest
julia> using ITensorBase: setname

julia> setname(named(2, :i), :j)
named(2, :j)
```

See also [`named`](@ref), [`name`](@ref).
"""
function setname end

"""
    nametype(type::Type)

The type of the name carried by a named type, such as a `Named` scalar type, a named
array type, or a named unit range type.

# Examples

```jldoctest
julia> using ITensorBase: nametype

julia> nametype(typeof(named(2, :i)))
Symbol
```

See also [`name`](@ref), [`unnamedtype`](@ref).
"""
function nametype end

"""
    unnamedtype(type::Type)

The type of the underlying (unnamed) value carried by a named type.

# Examples

```jldoctest
julia> using ITensorBase: unnamedtype

julia> unnamedtype(typeof(named(2, :i)))
Int64
```

See also [`unnamed`](@ref), [`nametype`](@ref).
"""
function unnamedtype end

# Minimal interface.
unnamed(i::Named) = i.value
name(i::Named) = i.name

"""
    named(value, name)

Attach `name` to `value`, pairing them into a single named object. On a scalar this produces
a `Named`. Arrays and unit ranges have their own more specific methods.

# Examples

```jldoctest
julia> named(2, :i)
named(2, :i)
```
"""
named(value, name) = Named(value, name)

# Derived interface.
setname(i::Named, name) = named(unnamed(i), name)
setvalue(i::Named, value) = named(value, name(i))

unnamedtype(::Type{<:Named{<:Any, Unnamed}}) where {Unnamed} = Unnamed
nametype(::Type{<:Named{Name}}) where {Name} = Name

# Traits.
isnamed(::Type{<:Named}) = true

function Base.:(==)(i1::Named, i2::Named)
    return name(i1) == name(i2) && unnamed(i1) == unnamed(i2)
end
# Hash under a literal tag plus the unnamed value and name. The tag is shared by an
# entire equality class (one for the scalar `Named`, one for all `AbstractNamedArray`
# types), not per concrete type, so `a == b => hash(a) == hash(b)` holds the way it
# does in Base (`[1, 2, 3] == 1:3` and they hash equally).
function hash_named(typetag::Symbol, x, h::UInt)
    h = hash(typetag, h)
    h = hash(unnamed(x), h)
    return hash(name(x), h)
end
Base.hash(i::Named, h::UInt) = hash_named(:Named, i, h)

function uniquename(rng::AbstractRNG, i::Named)
    return named(unnamed(i), uniquename(name(i)))
end

function Base.string(i::Named; kwargs...)
    return "named($(string(unnamed(i); kwargs...)), $(repr(name(i))))"
end
function Base.show(io::IO, i::Named)
    print(io, "named(", unnamed(i), ", ", repr(name(i)), ")")
    return nothing
end

# Integer interface, for the named-integer case `NamedInteger`.
Base.:-(i::NamedInteger) = setvalue(i, -unnamed(i))

## TODO: Support this, we need to define `NamedFloat`, `NamedReal`, `NamedNumber`, etc.
## This is used in `LinearAlgebra.norm`, for now we just overload that directly.
## Here, named numbers are treated as unitful, so multiplying them
## with unnamed numbers means the result inherits the name.
## function Base.:*(i1::NamedInteger, i2::Number)
##   return named(unnamed(i1) * i2, name(i1))
## end

Base.zero(i::NamedInteger) = setvalue(i, zero(unnamed(i)))
Base.one(i::NamedInteger) = setvalue(i, one(unnamed(i)))
Base.signbit(i::NamedInteger) = signbit(unnamed(i))
Base.unsigned(i::NamedInteger) = setvalue(i, unsigned(unnamed(i)))

# Used in bounds checking when indexing with named dimensions.
function Base.:<(i1::NamedInteger, i2::NamedInteger)
    name(i1) == name(i2) || throw(NameMismatch("Mismatched names $(name(i1)), $(name(i2))"))
    return unnamed(i1) < unnamed(i2)
end
