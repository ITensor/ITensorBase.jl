module ITensorBaseTensorKitExt

using ITensorBase: ITensorBase, NamedUnitRange
using TensorKit: ElementarySpace, dim

# ================================  Index over a native space  ==============================
# A native TensorKit space is stored directly as the axis value of a `NamedUnitRange`, so
# `Index(V)` (and `dual`/`conj` of one) round-trips through the named layer with the space
# intact. These terminal constructors short-circuit `to_range`; the element type is `Int`
# (the flat index positions), matching how the space presents to TensorAlgebra as an axis.
function ITensorBase.NamedUnitRange{Name}(unnamed::ElementarySpace, name) where {Name}
    return ITensorBase.NamedUnitRange{Name, Int, typeof(unnamed)}(unnamed, name)
end
function ITensorBase.NamedUnitRange(unnamed::ElementarySpace, name)
    return ITensorBase.NamedUnitRange{typeof(name), Int, typeof(unnamed)}(unnamed, name)
end

# `conj(index)` lowers to `named(conj(space), name)`, and `trivialrange` mints a fresh axis via
# `namedunitrange(space, name)`, so both must rebuild a `NamedUnitRange` over the space rather than
# fall through to the generic (array-shaped) `Named`. `named` delegates to `namedunitrange`, as the
# core `AbstractUnitRange` path does.
ITensorBase.namedunitrange(r::ElementarySpace, name) = ITensorBase.NamedUnitRange(r, name)
ITensorBase.named(r::ElementarySpace, name) = ITensorBase.namedunitrange(r, name)

# The flat length of a space-backed axis is its total (dense) dimension, so `size`/`length`
# of a `TensorMap`-backed ITensor report dense dimensions.
function Base.length(r::NamedUnitRange{<:Any, <:Any, <:ElementarySpace})
    return dim(ITensorBase.unnamed(r))
end

end
