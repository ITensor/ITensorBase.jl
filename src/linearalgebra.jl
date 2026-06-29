using LinearAlgebra: LinearAlgebra as LA

# We overload `LinearAlgebra.norm` because the LinearAlgebra.jl AbstractArray definition
# uses scalar indexing:
# https://github.com/JuliaLang/LinearAlgebra.jl/blob/3a4fdad7f608928ecb4b41e76b1e9ecacd058444/src/generic.jl#L575-L724
# which isn't friendly for named arrays wrapping GPU arrays.
# This implicitly helps with defining `LA.normalize[!]` as well (though note that
# uses `LinearAlgebra.rmul!` as well).
function LA.norm(a::AbstractNamedTensor, p::Real = 2; kwargs...)
    return LA.norm(unnamed(a), p; kwargs...)
end

# We overload these because the LinearAlgebra.jl AbstractArray definitions of `rmul!`,
# `lmul!`, `rdiv!`, and `ldiv!` use scalar indexing:
# https://github.com/JuliaLang/LinearAlgebra.jl/blob/3a4fdad7f608928ecb4b41e76b1e9ecacd058444/src/generic.jl#L266-L395
# which isn't friendly for named arrays wrapping GPU arrays.
for f! in [:mul!, :div!]
    lf! = Symbol(:l, f!)
    rf! = Symbol(:r, f!)
    @eval begin
        function LA.$rf!(a::AbstractNamedTensor, α::Number)
            LA.$rf!(unnamed(a), α)
            return a
        end
        function LA.$lf!(α::Number, a::AbstractNamedTensor)
            LA.$lf!(α, unnamed(a))
            return a
        end
    end
end

# We overload `LienarAlgebra.dot` because the LinearAlgebra.jl AbstractArray definition
# uses scalar indexing:
# https://github.com/JuliaLang/LinearAlgebra.jl/blob/3a4fdad7f608928ecb4b41e76b1e9ecacd058444/src/generic.jl#L919-L1009
# which isn't friendly for named arrays wrapping GPU arrays.
function LA.dot(a1::AbstractNamedTensor, a2::AbstractNamedTensor)
    return (conj(a1) * a2)[]
end
