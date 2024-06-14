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
        "Tokenizer" => "tokenizer.md",
        "Embeddings" => "embeddings.md",
        "Layers" => "layers.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/mashu/VDJDL.jl",
    devbranch="main",
)
