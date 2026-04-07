# Motion Based Sign Phonology

This repository presents the steps to learn phonological representations of sign language directly from video, without relying on gloss annotations or semantic supervision.

The core idea is to learn a phonological embedding space where signs are organized according to their motion properties.

## Method Overview

The method consists of two main stages:

### Stage 1: Motion-Aware Spatio-Temporal Encoding
Input: RGB video B×T×3×H×W

Transformed into a luminance-based motion representation:
- normalized luminance,
- temporal luminance differences,
- edge-based motion cues.

Processed using:
- 3D CNN backbone
- Transformer-based temporal attention

Auxiliary branch:
- Generates an MHI-like representation

### Stage 2: Phonological Embedding Space Construction
A lightweight projection head maps features into a low-dimensional embedding space

Training is performed using:
- Triplet loss
- Semi-hard negative mining

Output:
L2-normalized embeddings (e.g., 256-D)

## Datasets

The framework is evaluated on:
- Peruvian Sign Language (LSP) dataset
- American Sign Language (ASL) dataset


# Training
## Stage 1: Motion Encoder
Before running the script, make sure to configure:

- input and output paths,
- dataset directories,
- language-specific settings.

```python 1_trainingVideoEnbed.py```

## Stage 2: Metric Learning
Before running this stage, make sure to:

- set the path to the extracted features from Stage 1
- output paths,
- dataset directories,
- language-specific settings.

```python 2_trainingTripletSemiHard.py```

# citation
soon