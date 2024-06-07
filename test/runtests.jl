using Test
using VDJDL.Tokenizer: LabelEncoder, SequenceTokenizer, read_fasta
using VDJDL.Embeddings: PositionEncoding, make_position_encoding
using Flux

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
    @test decoded_matrix == [["A", "C"], ["B", "A"]]

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
    @test decoded_matrix == [['N', 'C'], ['A', 'G']]

    # Test case to trigger findfirst(isequal(unksym), alphabet)
    alphabet_with_unksym = ['N', 'A', 'C', 'G', 'T']
    tokenizer_with_existing_unksym = SequenceTokenizer(alphabet_with_unksym, 'N')
    @test tokenizer_with_existing_unksym.unkidx == 1  # 'N' should be found at index 1
    
    # Test Flux.trainable method
    @test length(Flux.trainable(tokenizer)) == 0
end

# Test the PositionEncoding struct and methods
@testset "PositionEncoding" begin
    # Test make_position_encoding function
    @testset "Make position encoding" begin
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

        # Explicitly test the return value
        @test isapprox(make_position_encoding(2, 2), expected_encoding, atol=1e-6)
    end

    # Test PositionEncoding constructor
    @testset "Constructor" begin
        dim_embedding = 4
        max_length = 20
        pe = PositionEncoding(dim_embedding, max_length)
        
        @test size(pe.weight) == (dim_embedding, max_length)
    end

    # Test application of PositionEncoding to arrays
    @testset "Apply to array" begin
        dim_embedding = 4
        max_length = 20
        pe = PositionEncoding(dim_embedding, max_length)
        
        # Create a dummy input array
        x = rand(Float32, dim_embedding, 10)
        encoding = pe(x)
        
        @test size(encoding) == (dim_embedding, 10)
    end

    # Test handling of sequence lengths
    @testset "Sequence length handling" begin
        dim_embedding = 4
        max_length = 20
        pe = PositionEncoding(dim_embedding, max_length)
        
        # Valid sequence length
        seq_length = 10
        encoding = pe(seq_length)
        @test size(encoding) == (dim_embedding, seq_length)
        
        # Invalid sequence length (greater than max_length)
        seq_length = 25
        @test_throws ErrorException pe(seq_length)
    end

    # Test Base.show method
    @testset "Base.show method" begin
        dim_embedding = 4
        max_length = 20
        pe = PositionEncoding(dim_embedding, max_length)
        
        io = IOBuffer()
        Base.show(io, pe)
        output = String(take!(io))
        
        @test output == "PositionEncoding($(dim_embedding))"
    end

    # Test Flux.trainable method
    @testset "Flux.trainable method" begin
        dim_embedding = 4
        max_length = 20
        pe = PositionEncoding(dim_embedding, max_length)
        
        trainables = Flux.trainable(pe)
        
        @test length(trainables) == 0
    end
end
