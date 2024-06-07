# VDJDL

[![Build Status](https://github.com/mashu/VDJDL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mashu/VDJDL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/github/mashu/VDJDL.jl/graph/badge.svg?token=ZSX40TP0HZ)](https://codecov.io/github/mashu/VDJDL.jl)

## Description

VDJDL is a Julia package specifically designed for deep learning tasks involving VDJ immunoreceptor sequences. It offers tools and functionalities for efficient training of deep learning models using the Flux library. VDJDL is tailored to support the unique challenges of working with VDJ recombination data, providing a robust framework for researchers and developers in computational biology and bioinformatics.

## Features

- **Label Encoding**: Convert categorical labels to numerical format.
- **Sequence Tokenization**: Tokenize DNA sequences for model input.
- **Position Embeddings**: Generate position encodings for sequence data.

## Installation

To install VDJDL, use the following command:

```julia
using Pkg
Pkg.add("https://github.com/mashu/VDJDL.jl")
```
