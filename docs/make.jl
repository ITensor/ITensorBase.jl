using ITensorBase: ITensorBase
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  ITensorBase, :DocTestSetup, :(using ITensorBase); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[ITensorBase],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensorBase.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/ITensorBase.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/ITensor/ITensorBase.jl", devbranch="main", push_preview=true
)
