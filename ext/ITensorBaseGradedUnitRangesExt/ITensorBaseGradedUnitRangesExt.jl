module ITensorBaseGradedUnitRangesExt

using GradedUnitRanges: AbstractGradedUnitRange
using ITensorBase: ITensorBase

# TODO: Replace with a more general functionality in
# `GradedUnitRanges`, like `isgraded`.
ITensorBase.hasqns(r::AbstractGradedUnitRange) = true

end
