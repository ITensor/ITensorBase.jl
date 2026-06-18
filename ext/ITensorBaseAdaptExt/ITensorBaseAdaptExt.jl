module ITensorBaseAdaptExt

using Adapt: Adapt, adapt
using ITensorBase: AbstractNamedDimsArray, denamed, dimnames, nameddims

function Adapt.adapt_structure(to, a::AbstractNamedDimsArray)
    return nameddims(adapt(to, denamed(a)), dimnames(a))
end

end
