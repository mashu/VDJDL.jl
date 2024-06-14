module Layers
    using Flux
    export Rezero

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
end