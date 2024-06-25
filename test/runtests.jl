using Test
using VDJDL.Tokenizer: LabelEncoder, SequenceTokenizer, read_fasta
using VDJDL.Embeddings: PositionEncoding, make_position_encoding
using VDJDL.Layers: Rezero, RotaryMultiHeadAttention
using VDJDL.VDJDistributions: ShiftedPoisson, ZISNB, MixtureZISNB, BiZISNB, loglikelihood, eps
using Flux
using FiniteDifferences
using Distributions
using Random

# Helper function to create a temporary FASTA file
function create_temp_fasta(content::String)
    filepath = tempname() * ".fasta"
    open(filepath, "w") do file
        write(file, content)
    end
    return filepath
end

# Test read_fasta function
@testset "read_fasta" begin
    fasta_content = """>seq1
    ATGCGT
    >seq2
    CGTACG
    """
    filepath = create_temp_fasta(fasta_content)
    records = read_fasta(filepath)
    @test records == [("seq1", "ATGCGT"), ("seq2", "CGTACG")]
end

 # Test LabelEncoder functionality
 @testset "LabelEncoder" begin
    labels = ["A", "B", "C"]
    encoder = LabelEncoder(labels)

    @test length(encoder) == 3
    @test encoder("A") == 1
    @test encoder("B") == 2
    @test encoder("C") == 3
    @test encoder(1) == "A"
    @test encoder(2) == "B"
    @test encoder(3) == "C"
    @test encoder(["A", "B", "C"]) == [1, 2, 3]
    @test encoder([1, 2, 3]) == ["A", "B", "C"]

    # Test show method for LabelEncoder
    io = IOBuffer()
    Base.show(io, encoder)
    output = String(take!(io))
    @test output == "LabelEncoder(length(labels)=3)"

    # Test AbstractVector{String} method for LabelEncoder
    labels_batch = ["A", "C"]
    encoded_batch = encoder(labels_batch)
    @test encoded_batch == [1, 3]

    # Test AbstractVector{Int} method for LabelEncoder
    indices_batch = [2, 3]
    decoded_batch = encoder(indices_batch)
    @test decoded_batch == ["B", "C"]

    # Test AbstractMatrix{Int} method for LabelEncoder
    indices_matrix = [1 2; 3 1]
    decoded_matrix = encoder(indices_matrix)
    @test decoded_matrix == ["A" "B"; "C" "A"]

    # Test Flux.trainable method
    @test length(Flux.trainable(encoder)) == 0
end

# Test SequenceTokenizer functionality
@testset "SequenceTokenizer" begin
    alphabet = ['A', 'C', 'G', 'T']
    unksym = 'N'
    tokenizer = SequenceTokenizer(alphabet, unksym)

    @test length(tokenizer) == 5  # Alphabet + unksym
    @test tokenizer('A') == 2  # 'A' is at index 2
    @test tokenizer('N') == 1  # 'N' is at index 1 (unksym)
    @test tokenizer('Z') == 1  # Unrecognized symbol should return unkidx
    @test tokenizer(['A', 'C', 'T']) == [2, 3, 5]
    @test tokenizer([1, 2, 3]) == ['N', 'A', 'C']
    @test tokenizer([[2, 3], [4, 5]]) == [['A', 'C'], ['G', 'T']]

    # Test show method for SequenceTokenizer
    io = IOBuffer()
    Base.show(io, tokenizer)
    output = String(take!(io))
    @test output == "SequenceTokenizer{Char}(length(alphabet)=5, unksym=N)"

    # Test AbstractVector{Vector{T}} method for SequenceTokenizer
    batch = [['A', 'C'], ['G', 'T']]
    encoded_batch = tokenizer(batch)
    @test size(encoded_batch) == (2, 2)
    @test encoded_batch == [2 4; 3 5]

    # Test AbstractVector{Vector{Int}} method for SequenceTokenizer
    indices_batch = [[1, 2], [3, 4]]
    decoded_batch = tokenizer(indices_batch)
    @test decoded_batch == [['N', 'A'], ['C', 'G']]

    # Test AbstractMatrix{Int} method for SequenceTokenizer
    indices_matrix = [1 2; 3 4]
    decoded_matrix = tokenizer(indices_matrix)
    @test decoded_matrix == ['N' 'A'; 'C' 'G']

    # Test case to trigger findfirst(isequal(unksym), alphabet)
    alphabet_with_unksym = ['N', 'A', 'C', 'G', 'T']
    tokenizer_with_existing_unksym = SequenceTokenizer(alphabet_with_unksym, 'N')
    @test tokenizer_with_existing_unksym.unkidx == 1  # 'N' should be found at index 1
    
    # Test Flux.trainable method
    @test length(Flux.trainable(tokenizer)) == 0
end

# Main testset for PositionEncoding
@testset "PositionEncoding" begin
    # Test make_position_encoding function
    dim_embedding = 4
    seq_length = 10
    encoding = make_position_encoding(dim_embedding, seq_length)
    
    @test size(encoding) == (dim_embedding, seq_length)
    
    for pos in 1:seq_length
        for row in 0:2:(dim_embedding - 1)
            denom = 1 / (10000^(row / dim_embedding))
            @test encoding[row + 1, pos] ≈ sin(pos * denom)
            @test encoding[row + 2, pos] ≈ cos(pos * denom)
        end
    end

    expected_encoding = Matrix{Float32}([
        0.841471 0.909297;
        0.540302 -0.416147
    ])

    @test isapprox(make_position_encoding(2, 2), expected_encoding, atol=1e-6)

    # Test PositionEncoding constructor
    max_length = 20
    pe = PositionEncoding(dim_embedding, max_length)
    @test size(pe.weight) == (dim_embedding, max_length)

    # Test application of PositionEncoding to arrays
    x = rand(Float32, dim_embedding, 10)
    encoding = pe(x)
    @test size(encoding) == (dim_embedding, 10)

    # Test handling of sequence lengths
    seq_length = 10
    encoding = pe(seq_length)
    @test size(encoding) == (dim_embedding, seq_length)
    
    seq_length = 25
    @test_throws ErrorException pe(seq_length)

    # Test Base.show method
    io = IOBuffer()
    Base.show(io, pe)
    output = String(take!(io))
    @test output == "PositionEncoding($(dim_embedding))"

    # Test Flux.trainable method
    trainables = Flux.trainable(pe)
    @test length(trainables) == 0
end

# Test ReZero
@testset "Rezero" begin
    # Create a Rezero layer with a custom alpha parameter
    alpha = [0.5]
    rezero = Rezero(alpha)
    
    @test rezero.alpha == alpha
    
    # Create a Rezero layer with default alpha parameter
    rezero = Rezero()
    
    @test rezero.alpha == [0f0]
    
    # Apply the Rezero layer to an input array
    x = rand(Float32, 4, 4)
    y = rezero(x)
    
    @test y == zeros(Float32, 4, 4)
    
    # Test Base.show method
    io = IOBuffer()
    Base.show(io, rezero)
    output = String(take!(io))
    
    @test output == "Rezero(alpha = Float32[0.0])"
    
    # Test Flux.trainable method
    trainables = Flux.trainable(rezero)
    
    @test length(trainables) == 1

    # Test gradients
    x = rand(Float32, 5)
    alpha = rand(Float32, 5)
    layer = Rezero(alpha)
    
    model = Chain(layer)
    loss(x) = sum(model(x))
    
    grads = Flux.gradient(() -> loss(x), Flux.params(model))[layer.alpha]
    
    function fd_loss(alpha)
        layer.alpha .= alpha  # Update layer's alpha with the new alpha
        return sum(layer(x))
    end
    
    fdm = FiniteDifferences.central_fdm(5, 1)
    fd_grads = FiniteDifferences.grad(fdm, fd_loss, layer.alpha)[1]

    # Test if gradients are approximately equal
    @test isapprox(grads, fd_grads, atol=1e-5)
end

@testset "RotaryMultiHeadAttention" begin
    # Test for RotaryMultiHeadAttention struct creation
    hidden_size = 256
    head_size = 64
    nheads = 8
    bias = true
    dropout_prob = 0.1

    rma = RotaryMultiHeadAttention(hidden_size, head_size, nheads; bias=bias, dropout_prob=dropout_prob)

    @test rma.hidden_size == hidden_size
    @test rma.head_size == head_size
    @test rma.nheads == nheads
    @test rma.bias == bias
    @test rma.dropout_prob == dropout_prob

    # Test for the custom string representation
    io = IOBuffer()
    rma = RotaryMultiHeadAttention(256, 64, 8)
    show(io, rma)
    output = String(take!(io))

    expected_output = "RotaryMultiHeadAttention(nheads=8, hidden_size=256, head_size=64, bias=false, dropout_prob=0.0)"
    @test output == expected_output
    # Test for the rotary multi-head attention mechanism
    hidden_size = 256
    head_size = 64
    nheads = 8
    rma = RotaryMultiHeadAttention(hidden_size, head_size, nheads)

    q_in = rand(Float32, hidden_size, 10, 32)  # Query input
    k_in = rand(Float32, hidden_size, 10, 32)  # Key input
    v_in = rand(Float32, hidden_size, 10, 32)  # Value input

    x, α = rma(q_in, k_in, v_in)

    @test size(x) == (hidden_size, 10, 32)
    @test size(α) == (10, 10, 1, 32)
end

# Test for ShiftedPoisson
@testset "ShiftedPoisson Tests" begin
    λ = 3.0
    shift = 2
    sp = ShiftedPoisson(λ, shift)

    @test isapprox(pdf(sp, 5), pdf(Poisson(λ), 5 - shift))
    @test isapprox(logpdf(sp, 5), logpdf(Poisson(λ), 5 - shift))

    Random.seed!(1234)
    rand_val = rand(sp)
    @test rand_val >= shift
end

# Test for ZISNB
@testset "ZISNB Tests" begin
    π = 0.3
    r = 2.0
    p = 0.5
    zisnb = ZISNB(π, r, p)

    Random.seed!(1234)
    rand_val = rand(zisnb)
    @test rand_val >= 0

    logpdf_val_zero = logpdf(zisnb, 0)
    logpdf_val_nonzero = logpdf(zisnb, 2)
    logpdf_val_negative = logpdf(zisnb, -1)
    @test isapprox(logpdf_val_zero, log(π))
    @test isapprox(logpdf_val_nonzero, log1p(-π) + logpdf(NegativeBinomial(r, p), 2 - 1))
    @test logpdf_val_negative == -Inf
end

# Test for MixtureZISNB
@testset "MixtureZISNB Tests" begin
    π = 0.3
    r = 2.0
    p = 0.5
    negative_min = -5
    negative_max = -1
    mixture_prob = 0.7
    zisnb = ZISNB(π, r, p)
    mzisnb = MixtureZISNB(zisnb, negative_min, negative_max, mixture_prob)
    
    Random.seed!(1234)
    rand_val = rand(mzisnb)
    @test rand_val >= negative_min && rand_val <= negative_max || rand_val >= 0
    
    # Test rand function for MixtureZISNB to cover DiscreteUniform component
    Random.seed!(1234)
    rand_vals = [rand(mzisnb) for _ in 1:100]
    count_negative = count(x -> x >= negative_min && x <= negative_max, rand_vals)
    count_nonnegative = count(x -> x >= 0, rand_vals)
    @test count_negative > 0
    @test count_nonnegative > 0
    
    # Valid positive value for logpdf
    @test isapprox(logpdf(mzisnb, 2), log(mixture_prob) + logpdf(zisnb, 2))

    # Valid negative value for logpdf
    @test isapprox(logpdf(mzisnb, -3), log1p(-mixture_prob) + logpdf(DiscreteUniform(negative_min, negative_max), -3))

    # Invalid negative value outside range for logpdf
    @test logpdf(mzisnb, -6) == -Inf

    # Test multiple rand values
    Random.seed!(1234)
    n = 10
    rand_vals = rand(Random.GLOBAL_RNG, mzisnb, n)

    @test length(rand_vals) == n
    @test all(((rand_vals .>= negative_min) .& (rand_vals .<= negative_max)) .| (rand_vals .>= 0))

    # Check that we have a mix of negative and non-negative values
    count_negative = count(x -> x >= negative_min && x <= negative_max, rand_vals)
    count_nonnegative = count(x -> x >= 0, rand_vals)
    @test count_negative > 0
    @test count_nonnegative > 0
end

# Test for BiZISNB
@testset "BiZISNB Tests" begin
    π_left = 0.3
    p_left = 0.5
    π_right = 0.4
    p_right = 0.6
    α = 0.1
    β = 0.2
    len = 10.0
    jt = BiZISNB(π_left, p_left, π_right, p_right, α, β, len)

    Random.seed!(1234)
    rand_val = rand(jt)
    n = 10
    rand_vals = rand(jt, n)
    @test length(rand_val) == 2
    @test length(rand_vals) == n

    mu = exp(α + β * len) + eps
    left_trimming = rand_val[1]
    right_trimming = rand_val[2]

    logpdf_val = logpdf(jt, rand_val)
    @test isapprox(logpdf_val, logpdf(ZISNB(π_left, mu, p_left), left_trimming) + logpdf(ZISNB(π_right, mu, p_right), right_trimming))

    loglikelihood_val = loglikelihood(jt, rand_val)
    @test isapprox(loglikelihood_val, logpdf_val)

end
