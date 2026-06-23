struct NamedArray{Name, UnnamedT, N, Unnamed <: AbstractArray{UnnamedT, N}} <:
    AbstractNamedArray{Name, UnnamedT, N}
    value::Unnamed
    name::Name
end

# Minimal interface.
unnamed(a::NamedArray) = a.value
name(a::NamedArray) = a.name
unnamedtype(::Type{<:NamedArray{<:Any, <:Any, <:Any, Unnamed}}) where {Unnamed} = Unnamed
