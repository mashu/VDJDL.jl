using VDJDL
using Documenter

DocMeta.setdocmeta!(VDJDL, :DocTestSetup, :(using VDJDL); recursive=true)

makedocs(;
    modules=[VDJDL],
    authors="Mateusz Kaduk <mateusz.kaduk@gmail.com> and contributors",
    sitename="VDJDL.jl",
    format=Documenter.HTML(;
        canonical="https://mashu.github.io/VDJDL.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    strict = false  # Disable strict mode to avoid errors on missing docs
)

deploydocs(;
    repo="github.com/mashu/VDJDL.jl",
    devbranch="main",
)
