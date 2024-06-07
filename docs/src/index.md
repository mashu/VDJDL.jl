```@meta
CurrentModule = VDJDL
```
# VDJ Deep Learning

[VDJDL](https://github.com/mashu/VDJDL.jl) (VDJ Deep Learning) is a Julia package specifically designed for deep learning tasks involving VDJ immunoreceptor sequences. It offers tools and functionalities for efficiently training deep learning models using the [Flux](https://fluxml.ai/Flux.jl/) library. VDJDL is tailored to support the unique challenges of working with VDJ recombination data, providing a robust framework for researchers and developers in computational biology and bioinformatics.

# Modules
- [Tokenizer](tokenizer.md)
- [Emebddings](embeddings.md)

# Quick Start

## Sequence Tokenizer

The Sequence Tokenizer provides a mapping between characters and integers, and vice versa.

```@example
using VDJDL: SequenceTokenizer

tokenizer = SequenceTokenizer(collect("ATGC"), '-')
tokenizer(collect.(["ATGC", "TGCTTG", "GGG"]))
```

## Positional Encoding
To use sine-cosine positional embeddings, the following Flux-compatible layer is implemented.

```@example
using VDJDL: PositionEncoding

emb_dim = 16
seq_length = 100
pe = PositionEncoding(emb_dim)
pe(seq_length)
```

```@autodocs
Modules = [VDJDL]
```
