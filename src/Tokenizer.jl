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
        LabelEncoder(labels::Vector{String})

    Encodes and decodes labels.

    # Arguments
    - `labels::Vector{String}`: A vector of labels to be encoded.

    # Returns
    - `LabelEncoder`: A `LabelEncoder` object.
    """
    struct LabelEncoder{T <: AbstractVector{String}}
        labels::T
        lookup::Dict{String, Int32}
        function LabelEncoder(labels::T) where T <: AbstractVector{String}
            lookup = Dict(label => Int32(idx) for (idx, label) in enumerate(labels))
            new{T}(labels, lookup)
        end
    end

    Base.length(encoder::LabelEncoder) = length(encoder.labels)

    function Base.show(io::IO, encoder::LabelEncoder)
        print(io, "LabelEncoder(length(labels)=$(length(encoder)))")
    end

    function (encoder::LabelEncoder)(label::String)
        haskey(encoder.lookup, label) ? encoder.lookup[label] : error("Label not found: $label")
    end

    function (encoder::LabelEncoder)(idx::Integer)
        encoder.labels[idx]
    end

    function (encoder::LabelEncoder)(x::A) where A <: AbstractArray
        map(i -> encoder(i), x)
    end

    Flux.@layer LabelEncoder
    Flux.trainable(encoder::LabelEncoder) = (;)

    """
        SequenceTokenizer(alphabet::Vector{T}, unksym::T) where T

    Tokenizes character sequences.

    # Arguments
    - `alphabet::Vector{T}`: A vector of symbols to be tokenized.
    - `unksym::T`: The symbol for unknown tokens.

    # Returns
    - `SequenceTokenizer{T}`: A `SequenceTokenizer` object.
    """ 
    struct SequenceTokenizer{T, V <: AbstractVector{T}}
        alphabet::V
        lookup::Dict{T, Int32}
        unksym::T
        unkidx::Int32

        function SequenceTokenizer(alphabet::V, unksym::T) where {T, V <: AbstractVector{T}}
            if !(unksym ∈ alphabet)
                alphabet = vcat(unksym, alphabet)
                unkidx = Int32(1)
            else
                unkidx = findfirst(isequal(unksym), alphabet)
            end
            lookup = Dict(x => Int32(idx) for (idx, x) in enumerate(alphabet))
            new{T, V}(alphabet, lookup, unksym, unkidx)
        end
    end

    Base.length(tokenizer::SequenceTokenizer) = length(tokenizer.alphabet)

    function Base.show(io::IO, tokenizer::SequenceTokenizer{T}) where T
        print(io, "SequenceTokenizer{$(T)}(length(alphabet)=$(length(tokenizer)), unksym=$(tokenizer.unksym))")
    end

    function (tokenizer::SequenceTokenizer{T})(token::T) where T
        haskey(tokenizer.lookup, token) ? tokenizer.lookup[token] : tokenizer.unkidx
    end

    function (tokenizer::SequenceTokenizer)(idx::Integer)
        tokenizer.alphabet[idx]
    end

    function (tokenizer::SequenceTokenizer{T})(x::A) where {T, A <: AbstractArray}
        map(i -> tokenizer(i), x)
    end

    function (tokenizer::SequenceTokenizer{T})(batch::A) where {T, A <: AbstractVector{<:AbstractVector{T}}}
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

    Flux.@layer SequenceTokenizer
    Flux.trainable(tokenizer::SequenceTokenizer) = (;)

end # module Tokenizer
