module Layers
    using Flux
    using NeuralAttentionlib: with_rotary_position_embedding
    export Rezero, RotaryMultiHeadAttention

    """
    Rezero layer implementation for Flux.jl. This layer multiplies the input by a learnable parameter alpha.
    Note this layer leads to higher loss value as comapred to LayerNorm given everything else constant in our experiments, but it is faster.

    # Fields
    - `alpha::AbstractVector`: The learnable parameter initialized by the user.

    # Constructor
    - `Rezero(init::AbstractVector)`: Create a new Rezero layer with the specified initial values for `alpha`.
    - `Rezero()`: Create a new Rezero layer with `alpha` initialized to zeros of type `Float32` with length 1.

    # Methods
    - `(a::Rezero)(x::AbstractArray)`: Apply the Rezero layer to the input array `x`.
    """
    struct Rezero{M<:AbstractVector}
        alpha::M
        """
        Create a new Rezero layer.

        # Arguments
        - `init::AbstractVector`: Initial values for the `alpha` parameter.

        # Returns
        - `Rezero`: A new instance of the Rezero layer.
        """
        function Rezero(init::M) where M<:AbstractVector
            new{M}(init)
        end
    end

    Flux.@layer Rezero

    """
    Create a new Rezero layer with `alpha` initialized to zeros of type `Float32` with length 1.

    # Returns
    - `Rezero`: A new instance of the Rezero layer.
    """
    Rezero() = Rezero(zeros(Float32, 1))

    """
    Apply the Rezero layer to the input array `x`.

    # Arguments
    - `x::AbstractArray`: The input array.

    # Returns
    - `AbstractArray`: The transformed array after applying the Rezero layer.
    """
    function (a::Rezero)(x::AbstractArray)
        return a.alpha .* x
    end

    """
    Show the Rezero layer in a human-readable format.

    # Arguments
    - `io::IO`: The I/O stream.
    - `m::Rezero`: The Rezero layer instance.
    """
    function Base.show(io::IO, m::Rezero)
        print(io, "Rezero(alpha = ", m.alpha, ")")
    end

    """
        struct RotaryMultiHeadAttention

    A struct representing a multi-head attention mechanism with rotary position embeddings.

    # Fields
    - `mha::MultiHeadAttention`: An instance of `MultiHeadAttention` from Flux.
    """
    struct RotaryMultiHeadAttention
        hidden_size::Int
        head_size::Int
        nheads::Int
        bias::Bool
        dropout_prob::Float64
        mha::MultiHeadAttention
    end

    Flux.@layer RotaryMultiHeadAttention

    """
        RotaryMultiHeadAttention(hidden_size::Int, head_size::Int, nheads::Int; bias::Bool=false, dropout_prob::Float64=0.0) -> RotaryMultiHeadAttention

    Create a `RotaryMultiHeadAttention` instance.

    # Arguments
    - `hidden_size::Int`: The size of the hidden layer.
    - `head_size::Int`: The size of each attention head.
    - `nheads::Int`: The number of attention heads.
    - `bias::Bool`: Whether to use bias in the attention layers (default: `false`).
    - `dropout_prob::Float64`: Dropout probability for the attention mechanism (default: `0.0`).

    # Returns
    - A `RotaryMultiHeadAttention` instance.
    """
    function RotaryMultiHeadAttention(hidden_size::Int, head_size::Int, nheads::Int; bias::Bool=false, dropout_prob::Float64=0.0)
        mha = MultiHeadAttention(hidden_size => head_size => hidden_size, nheads=nheads, bias=bias, dropout_prob=dropout_prob)
        return RotaryMultiHeadAttention(hidden_size, head_size, nheads, bias, dropout_prob, mha)
    end

    """
        (r::RotaryMultiHeadAttention)(q_in, k_in, v_in; bias=nothing, mask=nothing) -> Tuple

    Apply the rotary multi-head attention mechanism to the input queries, keys, and values.

    # Arguments
    - `r::RotaryMultiHeadAttention`: An instance of `RotaryMultiHeadAttention`.
    - `q_in`: Input queries.
    - `k_in`: Input keys.
    - `v_in`: Input values.
    - `bias`: Optional bias for the attention mechanism.
    - `mask`: Optional mask for the attention mechanism.

    # Returns
    - A tuple `(x, α)` where `x` is the output of the attention mechanism and `α` is the attention weights.
    """
    function (r::RotaryMultiHeadAttention)(q_in, k_in, v_in; bias=nothing, mask=nothing)
        q = with_rotary_position_embedding(r.mha.q_proj(q_in))
        k = with_rotary_position_embedding(r.mha.k_proj(k_in))
        v = r.mha.v_proj(v_in)
        x, α = NNlib.dot_product_attention(q, k, v, bias; mask, fdrop=r.mha.attn_drop)
        x = r.mha.out_proj((x))
        return x, α
    end

    """
        Base.show(io::IO, r::RotaryMultiHeadAttention)

    Custom string representation for `RotaryMultiHeadAttention`.

    # Arguments
    - `io::IO`: The IO stream.
    - `r::RotaryMultiHeadAttention`: An instance of `RotaryMultiHeadAttention`.

    # Example
    ```julia
    r = RotaryMultiHeadAttention(256, 64, 8)
    println(r)
    ```
    """
    function Base.show(io::IO, r::RotaryMultiHeadAttention)
        print(io, "RotaryMultiHeadAttention(")
        print(io, "nheads=$(r.nheads), ")
        print(io, "hidden_size=$(r.hidden_size), ")
        print(io, "head_size=$(r.head_size), ")
        print(io, "bias=$(r.bias), ")
        print(io, "dropout_prob=$(r.dropout_prob))")
    end
end