module Embeddings
    using Flux

    export PositionEncoding, make_position_encoding

    """
        PositionEncoding{W <: AbstractArray}(weight::W)

    Represents positional encodings for sequence data.

    # Arguments
    - `weight::W`: The weight matrix for positional encodings.
    """
    struct PositionEncoding{W <: AbstractArray}
        weight::W
    end

    # Define the functor for the PositionEncoding structure
    Flux.@layer PositionEncoding
    Flux.trainable(m::PositionEncoding) = (;)

    """
        PositionEncoding(dim_embedding::Int, max_length::Int=1000)

    Constructs a PositionEncoding object with specified embedding dimension and maximum sequence length.

    # Arguments
    - `dim_embedding::Int`: The dimension of the embedding.
    - `max_length::Int`: The maximum length of the sequence. Defaults to 1000.

    # Returns
    - `PositionEncoding`: A PositionEncoding object.
    """
    function PositionEncoding(dim_embedding::Int, max_length::Int=1000)
        W = make_position_encoding(dim_embedding, max_length)
        PositionEncoding(W)
    end

    """
        make_position_encoding(dim_embedding::Int, seq_length::Int, n::Int=10000)

    Generates a positional encoding matrix.

    # Arguments
    - `dim_embedding::Int`: The dimension of the embedding.
    - `seq_length::Int`: The length of the sequence.
    - `n::Int`: A scaling factor. Defaults to 10000.

    # Returns
    - `Matrix{Float32}`: A matrix containing positional encodings.
    """
    function make_position_encoding(dim_embedding::Int, seq_length::Int, n::Int=10000)
        encoding = Matrix{Float32}(undef, dim_embedding, seq_length)
        for pos in 1:seq_length
            for row in 0:2:(dim_embedding - 1)
                denom = 1 / (n^(row / dim_embedding))
                encoding[row + 1, pos] = sin(pos * denom)
                encoding[row + 2, pos] = cos(pos * denom)
            end
        end
        return encoding
    end

    """
        (pe::PositionEncoding)(x::AbstractArray)

    Applies positional encoding to the input array.

    # Arguments
    - `x::AbstractArray`: The input array.

    # Returns
    - The positional encoding for the input array.
    """
    (pe::PositionEncoding)(x::AbstractArray) = (pe::PositionEncoding)(size(x, 2))

    """
        (pe::PositionEncoding)(seq_length::Int)

    Retrieves the positional encoding for a specific sequence length.

    # Arguments
    - `seq_length::Int`: The length of the sequence.

    # Returns
    - A view of the positional encoding matrix for the given sequence length.
    """
    function (pe::PositionEncoding)(seq_length::Int)
        max_length = size(pe.weight, 2)
        if seq_length > max_length
            error("sequence length of $seq_length exceeds maximum position encoding length of $max_length")
        end
        view(pe.weight, :, Base.OneTo(seq_length))
    end

    """
        Base.show(io::IO, pe::PositionEncoding)

    Displays the PositionEncoding object.

    # Arguments
    - `io::IO`: The I/O stream.
    - `pe::PositionEncoding`: The PositionEncoding object.
    """
    function Base.show(io::IO, pe::PositionEncoding)
        print(io, "PositionEncoding($(size(pe.weight, 1)))")
    end

end
