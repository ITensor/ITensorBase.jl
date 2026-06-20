struct NamedUnitRange{Name, DenamedT <: Integer, Denamed <: AbstractUnitRange{DenamedT}} <:
    AbstractNamedUnitRange{Name, DenamedT}
    value::Denamed
    name::Name
end

# Minimal interface.
denamed(i::NamedUnitRange) = i.value
name(i::NamedUnitRange) = i.name
denamedtype(::Type{<:NamedUnitRange{<:Any, <:Any, Denamed}}) where {Denamed} = Denamed
