using Literate: Literate
using ITensorBase: ITensorBase

Literate.markdown(
  joinpath(pkgdir(ITensorBase), "examples", "README.jl"),
  joinpath(pkgdir(ITensorBase));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
