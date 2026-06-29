using Documenter: Documenter, DocMeta, deploydocs, makedocs
using ITensorBase
using ITensorFormatter: ITensorFormatter

# `using ITensorBase` (rather than `using ITensorBase: ITensorBase`) binds the exported
# names in `Main`. The tensor `show` qualifies the type relative to the active module, so
# without it the doctests and `@example` blocks render `ITensorBase.NamedTensor` instead
# of `NamedTensor`.
DocMeta.setdocmeta!(ITensorBase, :DocTestSetup, :(using ITensorBase); recursive = true)

ITensorFormatter.make_index!(pkgdir(ITensorBase))

makedocs(;
    modules = [ITensorBase],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "ITensorBase.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/ITensorBase.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = [
        "Home" => "index.md",
        "User Interface" => "user_interface.md",
        "Developer Interface" => "dev_interface.md",
        "Reference" => "reference.md",
    ]
)

deploydocs(;
    repo = "github.com/ITensor/ITensorBase.jl",
    devbranch = "main",
    push_preview = true
)
