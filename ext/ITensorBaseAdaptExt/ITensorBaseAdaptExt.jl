module ITensorBaseAdaptExt

using Adapt: Adapt, adapt
using ITensorBase: AbstractITensor, denamed, dimnames, nameddims

function Adapt.adapt_structure(to, a::AbstractITensor)
    return nameddims(adapt(to, denamed(a)), dimnames(a))
end

end
