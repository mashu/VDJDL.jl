module Embeddings
    using Flux

    struct PositionEncoding{W <: AbstractArray}
        weight::W
    end

    Flux.@functor PositionEncoding
    Flux.trainable(m::PositionEncoding) = (;)

    function PositionEncoding(dim_embedding::Int, max_length::Int=1000)
        W = make_position_encoding(dim_embedding, max_length)
        PositionEncoding(W)
    end

    function make_position_encoding(dim_embedding::Int, seq_length::Int, n::Int=10000)
        encoding = Matrix{Float32}(undef, dim_embedding, seq_length)
        for pos in 1:seq_length
            for row in 0:2:(dim_embedding - 1)
                denom = 1/(n^(row/dim_embedding))
                encoding[row + 1, pos] = sin(pos * denom)
                encoding[row + 2, pos] = cos(pos * denom)
            end
        end
        encoding    
    end

    (pe::PositionEncoding)(x::AbstractArray) = (pe::PositionEncoding)(size(x, 2))
    function (pe::PositionEncoding)(seq_length::Int)
        max_length = size(pe.weight, 2)
        if seq_length > max_length
            error("sequence length of $seq_length exceeds maximum position encoding length of $max_length")
        end
        view(pe.weight, :, Base.OneTo(seq_length))
    end

    function Base.show(io::IO, pe::PositionEncoding)
        print(io, "PositionEncoding($(size(pe.weight, 1)))")
    end

    export PositionEncoding
end