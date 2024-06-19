module VDJDL
    include("Tokenizer.jl")
    include("Embeddings.jl")
    include("Layers.jl")
    include("VDJDistributions.jl")
    using .Tokenizer
    using .Embeddings
    using .Layers
    using .VDJDistributions
end # module VDJDL

