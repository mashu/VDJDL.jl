using Test
using VDJDL.Tokenizer: LabelEncoder, SequenceTokenizer, read_fasta
using VDJDL.Embeddings: PositionEncoding, make_position_encoding

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

# Test LabelEncoder
@testset "LabelEncoder" begin
    labels = ["cat", "dog", "mouse"]
    encoder = LabelEncoder(labels)

    # Test length
    @test length(encoder) == 3

    # Test encoding single label
    @test encoder("cat") == 1
    @test encoder("dog") == 2

    # Test encoding multiple labels
    @test encoder(["cat", "mouse"]) == [1, 3]

    # Test decoding single index
    @test encoder(1) == "cat"
    @test encoder(3) == "mouse"

    # Test decoding multiple indices
    @test encoder([1, 2]) == ["cat", "dog"]

    # Test handling of unknown labels
    @test_throws ErrorException encoder("elephant")
end

# Test SequenceTokenizer
@testset "SequenceTokenizer" begin
    alphabet = ['A', 'T', 'C', 'G']
    unksym = 'N'
    tokenizer = SequenceTokenizer(alphabet, unksym)

    # Test length
    @test length(tokenizer) == 5

    # Test tokenizing single token
    @test tokenizer('A') == 2
    @test tokenizer('G') == 5
    @test tokenizer('N') == 1

    # Test tokenizing multiple tokens
    @test tokenizer(['A', 'C', 'N']) == [2, 4, 1]

    # Test decoding single index
    @test tokenizer(2) == 'A'
    @test tokenizer(4) == 'C'

    # Test decoding multiple indices
    @test tokenizer([2, 5]) == ['A', 'G']

    # Test handling of sequences
    sequences = [['A', 'T', 'C'], ['G', 'C', 'A']]
    @test tokenizer(sequences) == [2 5; 3 4; 4 2]

    # Test handling of unknown symbols
    sequences_with_unknown = [['A', 'T', 'X'], ['G', 'C', 'A']]
    @test tokenizer(sequences_with_unknown) == [2 5; 3 4; 1 2]
end

# Test the PositionEncoding struct and methods
@testset "PositionEncoding" begin
    # Test make_position_encoding function
    @testset "make_position_encoding" begin
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
    end

    # Test PositionEncoding constructor
    @testset "constructor" begin
        dim_embedding = 4
        max_length = 20
        pe = PositionEncoding(dim_embedding, max_length)
        
        @test size(pe.weight) == (dim_embedding, max_length)
    end

    # Test application of PositionEncoding to arrays
    @testset "apply to array" begin
        dim_embedding = 4
        max_length = 20
        pe = PositionEncoding(dim_embedding, max_length)
        
        # Create a dummy input array
        x = rand(Float32, dim_embedding, 10)
        encoding = pe(x)
        
        @test size(encoding) == (dim_embedding, 10)
    end

    # Test handling of sequence lengths
    @testset "sequence length handling" begin
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
end
