# File: README.md

# HMM Variant Detector

A Hidden Markov Model-based package for detecting A->G variants in NGS sequencing data.

## Features

- 2-state, 2-channel HMM for variant detection
- Baum-Welch algorithm for parameter estimation
- Forward algorithm for likelihood computation
- Viterbi algorithm for state sequence decoding
- Pure NumPy implementation for performance

## Wishlist

- additional hidden states, eg, mapQ, PHRED, strand bias, etc
- bed output
- igv visualizations

## Installation

```bash
pip install hmm-variant-detector
```

## Quick Start

```python
from hmm_variant_detector import VariantHMM
import numpy as np

# Create sample data (reference counts, alternate counts)
observations = np.array([[10, 0], [8, 2], [5, 5], [2, 8], [0, 10]])

# Initialize and train HMM
hmm = VariantHMM()
hmm.train(observations, max_iterations=100)

# Decode variant positions
states = hmm.decode(observations)
likelihood = hmm.forward(observations)

print(f"Predicted states: {states}")
print(f"Log likelihood: {likelihood}")
```

## License

MIT License
