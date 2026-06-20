struct NamedArray{Name, DenamedT, N, Denamed <: AbstractArray{DenamedT, N}} <:
    AbstractNamedArray{Name, DenamedT, N}
    value::Denamed
    name::Name
end

# Minimal interface.
denamed(a::NamedArray) = a.value
name(a::NamedArray) = a.name
denamedtype(::Type{<:NamedArray{<:Any, <:Any, <:Any, Denamed}}) where {Denamed} = Denamed
