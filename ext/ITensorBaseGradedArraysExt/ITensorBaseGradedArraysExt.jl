module ITensorBaseGradedArraysExt

using GradedArrays: AbstractGradedUnitRange
using ITensorBase: ITensorBase

# TODO: Replace with a more general functionality in
# `GradedArrays`, like `isgraded`.
ITensorBase.hasqns(r::AbstractGradedUnitRange) = true

end
