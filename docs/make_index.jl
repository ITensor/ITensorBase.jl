using Literate: Literate
using ITensorBase: ITensorBase

Literate.markdown(
  joinpath(pkgdir(ITensorBase), "examples", "README.jl"),
  joinpath(pkgdir(ITensorBase), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)
