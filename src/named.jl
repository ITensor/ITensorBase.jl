# A named value is a tagged scalar: it pairs an underlying value with a name. The
# named-integer case `NamedInteger` is used for index values and array
# sizes. `Named` is standalone (not `<: Integer` or `<: Number`): mixed
# named/unnamed arithmetic and operations like `i1 * i2` are not cleanly definable
# under the numeric contract, and inherited fallbacks risk silently dropping the
# name. So it supplies the integer-like surface it needs directly.
struct Named{Name, Unnamed}
    unnamed::Unnamed
    name::Name
end

# A named integer is just the integer case of `Named`. The alias gives it a
# readable name for dispatch (`NamedInteger`, `NamedInteger{IndexName}`, ...)
# without being a separate type.
const NamedInteger{Name, Unnamed <: Integer} = Named{Name, Unnamed}

"""
    name(a)

The name attached to a named object `a`, such as a `Named` scalar, a named array, or a
named unit range. `name` recovers the name, [`unnamed`](@ref) recovers the value.

# Examples

```jldoctest
julia> using ITensorBase: Named, name

julia> name(Named(2, :i))
:i
```

See also [`unnamed`](@ref), [`setname`](@ref).
"""
function name end

"""
    unnamed(a)

The underlying value of a named object `a`, with its name stripped off. [`name`](@ref)
recovers the name, `unnamed` recovers the value. On an [`AbstractNamedTensor`](@ref) it
returns the underlying unnamed array.

# Examples

```jldoctest
julia> using ITensorBase: Named, unnamed

julia> unnamed(Named(2, :i))
2
```

See also [`name`](@ref).
"""
function unnamed end

"""
    setname(a, name)

Return a copy of the named object `a` with its name replaced by `name`, keeping the
underlying value unchanged.

# Examples

```jldoctest
julia> using ITensorBase: Named, setname

julia> setname(Named(2, :i), :j)
Named(2, :j)
```

See also [`name`](@ref).
"""
function setname end

"""
    nametype(type::Type)
    nametype(a::AbstractNamedTensor)
    nametype(type::Type{<:AbstractNamedTensor})

The type of the name carried by a named object. For a `Named` scalar type, a named array
type, or a named unit range type this is the type of its single name; for a named tensor it
is the type of an individual dimension name. The primary methods dispatch on the type, and
`nametype(a::AbstractNamedTensor)` forwards to `nametype(typeof(a))`. A named tensor type
that does not fix its dimension-name flavor (such as the unparameterized `NamedTensor`)
returns `Any`, the same way `eltype(Array)` is `Any`.

# Examples

```jldoctest
julia> using ITensorBase: Named

julia> nametype(typeof(Named(2, :i)))
Symbol
```

```jldoctest
julia> a = NamedTensor(zeros(2, 3), (:i, :j));

julia> nametype(a)
Symbol

julia> nametype(typeof(a))
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
julia> using ITensorBase: Named, unnamedtype

julia> unnamedtype(typeof(Named(2, :i)))
Int64
```

See also [`unnamed`](@ref), [`nametype`](@ref).
"""
function unnamedtype end

# Minimal interface.
unnamed(i::Named) = i.unnamed
name(i::Named) = i.name

# Attach a name to a value whose type is only known at runtime, picking the named type that
# matches its shape. Call sites that know what they are building use the constructor
# (`Named`, `NamedUnitRange`, `NamedArray`, `NamedColon`) directly; the per-type methods live
# alongside those types.
to_named(value, name) = Named(value, name)

# Derived interface.
setname(i::Named, name) = Named(unnamed(i), name)
setunnamed(i::Named, unnamed) = Named(unnamed, name(i))

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
    return Named(unnamed(i), uniquename(name(i)))
end

function Base.string(i::Named; kwargs...)
    return "Named($(string(unnamed(i); kwargs...)), $(repr(name(i))))"
end
function Base.show(io::IO, i::Named)
    print(io, "Named(", unnamed(i), ", ", repr(name(i)), ")")
    return nothing
end

# Integer interface, for the named-integer case `NamedInteger`.
Base.:-(i::NamedInteger) = setunnamed(i, -unnamed(i))

## TODO: Support this, we need to define `NamedFloat`, `NamedReal`, `NamedNumber`, etc.
## This is used in `LinearAlgebra.norm`, for now we just overload that directly.
## Here, named numbers are treated as unitful, so multiplying them
## with unnamed numbers means the result inherits the name.
## function Base.:*(i1::NamedInteger, i2::Number)
##   return Named(unnamed(i1) * i2, name(i1))
## end

Base.zero(i::NamedInteger) = setunnamed(i, zero(unnamed(i)))
Base.one(i::NamedInteger) = setunnamed(i, one(unnamed(i)))
Base.signbit(i::NamedInteger) = signbit(unnamed(i))
Base.unsigned(i::NamedInteger) = setunnamed(i, unsigned(unnamed(i)))

# Used in bounds checking when indexing with named dimensions.
function Base.:<(i1::NamedInteger, i2::NamedInteger)
    name(i1) == name(i2) || throw(NameMismatch("Mismatched names $(name(i1)), $(name(i2))"))
    return unnamed(i1) < unnamed(i2)
end
