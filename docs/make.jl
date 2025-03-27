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
    canonical="https://itensor.github.io/ITensorBase.jl",
    edit_link="main",
    assets=["assets/favicon.ico", "assets/extras.css"],
  ),
  pages=["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo="github.com/ITensor/ITensorBase.jl", devbranch="main", push_preview=true
)
