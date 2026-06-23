module ITensorBaseAdaptExt

using Adapt: Adapt, adapt
using ITensorBase: AbstractITensor, dimnames, nameddims, unnamed

function Adapt.adapt_structure(to, a::AbstractITensor)
    return nameddims(adapt(to, unnamed(a)), dimnames(a))
end

end
