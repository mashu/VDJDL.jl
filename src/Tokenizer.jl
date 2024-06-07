module Tokenizer
    using Flux
    using FASTX

    export LabelEncoder, SequenceTokenizer, read_fasta

    """
        read_fasta(filepath::String)

    Reads a FASTA file and returns a vector of tuples containing the description and sequence of each record.

    # Arguments
    - `filepath::String`: The path to the FASTA file.

    # Returns
    - `Vector{Tuple{String, String}}`: A vector of tuples with the description and sequence.
    """
    function read_fasta(filepath::String)
        records = Vector{Tuple{String, String}}()
        for record in open(FASTA.Reader, filepath)
            push!(records, (FASTA.description(record), FASTA.sequence(record)))
        end
        return records
    end

    """
        struct LabelEncoder

    Encodes and decodes labels for machine learning tasks.
    """
    struct LabelEncoder
        labels::Vector{String}
        lookup::Dict{String, Int32}

        function LabelEncoder(labels::Vector{String})
            lookup = Dict(label => Int32(idx) for (idx, label) in enumerate(labels))
            new(labels, lookup)
        end
    end

    Base.length(encoder::LabelEncoder) = length(encoder.labels)

    function Base.show(io::IO, encoder::LabelEncoder)
        print(io, "LabelEncoder(length(labels)=$(length(encoder)))")
    end

    function (encoder::LabelEncoder)(label::String)
        haskey(encoder.lookup, label) ? encoder.lookup[label] : error("Label not found: $label")
    end

    function (encoder::LabelEncoder)(idx::Int)
        encoder.labels[idx]
    end

    function (encoder::LabelEncoder)(batch::AbstractVector{String})
        map(l -> encoder(l), batch)
    end

    function (encoder::LabelEncoder)(batch::AbstractVector{Int})
        map(i -> encoder(i), batch)
    end

    function (encoder::LabelEncoder)(batch::AbstractMatrix{Int})
        map(i -> encoder(i), eachcol(batch))
    end

    Flux.@layer LabelEncoder
    Flux.trainable(encoder::LabelEncoder) = (;)

    """
        struct SequenceTokenizer{T}

    Tokenizes sequences for NLP tasks.
    """
    struct SequenceTokenizer{T}
        alphabet::Vector{T}
        lookup::Dict{T, Int32}
        unksym::T
        unkidx::Int32

        function SequenceTokenizer(alphabet::Vector{T}, unksym::T) where T
            if !(unksym ∈ alphabet)
                pushfirst!(alphabet, unksym)
                unkidx = Int32(1)
            else
                unkidx = findfirst(isequal(unksym), alphabet)
            end
            lookup = Dict(x => idx for (idx, x) in enumerate(alphabet))
            new{T}(alphabet, lookup, unksym, unkidx)
        end
    end

    Base.length(tokenizer::SequenceTokenizer) = length(tokenizer.alphabet)

    function Base.show(io::IO, tokenizer::SequenceTokenizer)
        T = eltype(tokenizer.alphabet)
        print(io, "SequenceTokenizer{$(T)}(length(alphabet)=$(length(tokenizer)), unksym=$(tokenizer.unksym))")
    end

    function (tokenizer::SequenceTokenizer{T})(token::T) where T
        haskey(tokenizer.lookup, token) ? tokenizer.lookup[token] : tokenizer.unkidx
    end

    function (tokenizer::SequenceTokenizer{T})(tokens::AbstractVector{T}) where T
        map(t -> tokenizer(t), tokens)
    end

    function (tokenizer::SequenceTokenizer)(idx::Int)
        tokenizer.alphabet[idx]
    end

    function (tokenizer::SequenceTokenizer)(idxs::AbstractVector{Int})
        map(i -> tokenizer(i), idxs)
    end

    function (tokenizer::SequenceTokenizer)(batch::AbstractVector{Vector{T}}) where T
        lengths = map(length, batch)
        max_length = maximum(lengths)
        indices = fill(tokenizer.unkidx, max_length, length(batch))
        for j in eachindex(batch)
            local seq = batch[j]
            for i in eachindex(seq)
                @inbounds indices[i, j] = tokenizer(seq[i])
            end
        end
        indices
    end

    function (tokenizer::SequenceTokenizer)(batch::AbstractVector{Vector{Int}})
        map(seq -> map(i -> tokenizer(i), seq), batch)
    end

    function (tokenizer::SequenceTokenizer)(batch::AbstractMatrix{Int})
        map(seq -> map(i -> tokenizer(i), seq), eachcol(batch))
    end

    Flux.@layer SequenceTokenizer
    Flux.trainable(tokenizer::SequenceTokenizer) = (;)

end # module Tokenizer
