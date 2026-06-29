module ITensorBaseAdaptExt

using Adapt: Adapt, adapt
using ITensorBase: AbstractNamedTensor, dimnames, nameddims, unnamed

function Adapt.adapt_structure(to, a::AbstractNamedTensor)
    return nameddims(adapt(to, unnamed(a)), dimnames(a))
end

end
