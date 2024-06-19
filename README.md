# VDJDL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mashu.github.io/VDJDL.jl/dev/)
[![Build Status](https://github.com/mashu/VDJDL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mashu/VDJDL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/github/mashu/VDJDL.jl/graph/badge.svg?token=ZSX40TP0HZ)](https://codecov.io/github/mashu/VDJDL.jl)

## Description

VDJDL is a Julia package specifically designed for deep learning tasks involving VDJ immunoreceptor sequences. It offers tools and functionalities for efficient training of deep learning models using the Flux library. VDJDL is tailored to support the unique challenges of working with VDJ recombination data, providing a robust framework for researchers and developers in computational biology and bioinformatics.

## Features

### Data representation
- **Label Encoding**: Converts categorical labels into numerical format.
- **Sequence Tokenization**: Tokenizes DNA sequences, preparing them for input into models.

### Layers
- **Position Encoding**: Implements absolute positional sine-cosine embeddings, providing positional information to sequences ([Attetnion Is All You Need](https://doi.org/10.48550/arXiv.1706.03762))
- **Rotary MultiHeadAttention**: Implements RoPE positional MultiHeadAttention layer as a wrapper of [NeuralAttentionlib.jl](https://github.com/chengchingwen/NeuralAttentionlib.jl) function ([RoFormer: Enhanced Transformer with Rotary Position Embedding](https://doi.org/10.48550/arXiv.2104.09864))
- **Rezero**: A normalization layer featuring a single learnable scaling parameter ([ReZero is All You Need: Fast Convergence at Large Depth](https://doi.org/10.48550/arXiv.2003.04887))

### Distributions
- ZISNB (Zero Inflated Shifted Negative Binomial, DiscreteUniform) to handle zeros separetly from NB
- MixtureZISNB (ZISNB, DiscreteUniform) to handle V and J trimmings with negative counts
- BiZISNB (ZISNB, ZISNB, length) to handle conditionally on length, non-negative trimmings of D on both 5' and 3' ends

## Installation

To install VDJDL, use the following command:

```julia
using Pkg
Pkg.add("https://github.com/mashu/VDJDL.jl")
```
