module ITensorBaseAdaptExt

using Adapt: Adapt, adapt
using ITensorBase: AbstractNamedTensor, NamedTensor, names, unnamed

function Adapt.adapt_structure(to, a::AbstractNamedTensor)
    return NamedTensor(adapt(to, unnamed(a)), names(a))
end

end
