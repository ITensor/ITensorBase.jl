abstract type AbstractName end
# TODO: Decide if this is a good definition, probably not.
# name(n::AbstractName) = throw(MethodError(name, Tuple{typeof(n)}))
Base.getindex(n::AbstractName, I) = named(I, name(n))

struct Name{Value} <: AbstractName
    value::Value
end
name(n::Name) = n.value
name_type(n::Name) = name_type(typeof(n))
name_type(n::Type{<:Name{Value}}) where {Value} = Value
function uniquename(rng::AbstractRNG, type::Type{<:Name}; kwargs...)
    return Name(uniquename(rng, name_type(type); kwargs...))
end

"""
    @names x y ...
    @names x[1:3] y[1:3, 2:4] ...

Short-hand notation for constructing "named symbols", i.e. objects that can be used as names.
In other words, the following expressions are equivalent:

```julia
x, y, z = @names x y z
x, y, z = Name.((:x, :y, :z))
```
"""
macro names(exs...)
    length(exs) == 1 && return esc(:($(_parse_name(only(exs)))))
    syms_exs = map(_parse_name, exs)
    return esc(:(($(syms_exs...),)))
end

_parse_name(ex::Symbol) = :($(Name(ex)))
function _parse_name(ex)
    Meta.isexpr(ex, :ref) || throw(ArgumentError("invalid @names expression: $ex"))
    length(ex.args) > 1 ||
        throw(
        ArgumentError(
            "@names indexing expression requires at least one set of indices"
        )
    )
    sym = QuoteNode(first(ex.args))
    if length(ex.args) == 2
        return :([$Name(Symbol($sym, :_, x)) for x in $(ex.args[2])])
    else
        return :(
            [
                $Name(Symbol($sym, Iterators.flatmap(y -> (:_, y), x)...)) for
                    x in Iterators.product($(ex.args[2:end]...))
            ]
        )
    end
end

struct NameMismatch <: Exception
    message::String
end
NameMismatch() = NameMismatch("")
