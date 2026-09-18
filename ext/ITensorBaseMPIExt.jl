module ITensorBaseMPIExt

using ITensorBase: AbstractNamedArray, AbstractNamedTensor, unnamed
using MPI: MPI

MPI.Buffer(a::AbstractNamedArray) = MPI.Buffer(unnamed(a))
MPI.Buffer(a::AbstractNamedTensor) = MPI.Buffer(unnamed(a))

end
