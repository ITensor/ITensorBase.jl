# Lazy broadcasting.
struct LazyNamedTensorStyle <: Base.Broadcast.AbstractArrayStyle{Any} end
function Broadcast.broadcasted(::LazyNamedTensorStyle, f, as...)
    return error("Arbitrary broadcasting not supported for LazyNamedTensor.")
end
# Linear operations.
Broadcast.broadcasted(::LazyNamedTensorStyle, ::typeof(+), a1, a2) = a1 + a2
Broadcast.broadcasted(::LazyNamedTensorStyle, ::typeof(-), a1, a2) = a1 - a2
Broadcast.broadcasted(::LazyNamedTensorStyle, ::typeof(*), c::Number, a) = c * a
Broadcast.broadcasted(::LazyNamedTensorStyle, ::typeof(*), a, c::Number) = a * c
Broadcast.broadcasted(::LazyNamedTensorStyle, ::typeof(*), a::Number, b::Number) = a * b
Broadcast.broadcasted(::LazyNamedTensorStyle, ::typeof(/), a, c::Number) = a / c
Broadcast.broadcasted(::LazyNamedTensorStyle, ::typeof(-), a) = -a
