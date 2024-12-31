# Objective

This work aims to reimplement the architecture described in the paper titled [A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multi-Pitch Estimation](https://arxiv.org/abs/2203.09893).

The goal is to automatically transcribe notes and estimate multiple pitches (multi-pitch) in an instrument-agnostic manner, meaning the model can handle various types of instruments without requiring specific adaptations.

## Model Summary

The model is a lightweight architecture based on neural networks and uses the following techniques:

- **Harmonic Stacking** to represent audio data in a compact and informative way.
- The model’s three output posteriorgrams are time-frequency matrices:
  - **Yp**: A raw multi-pitch estimate capturing the present frequencies.
  - **Yn**: A quantized representation to detect note events.
  - **Yo**: An onset estimation (note beginnings).

<p align="center">
  <img src="architecture.png" alt="Architecture du Modèle" width="500">
</p>

## Database Choice

In the original paper, the architecture was trained on five datasets (seven in total including test sets). For this project, we decided to focus solely on the **GuitarSet** dataset for the following reasons:

- **Diverse Characteristics**: GuitarSet contains both monophonic and polyphonic audio recordings, making it ideal for evaluating an instrument-agnostic model.
- **Rich Annotations**: This dataset provides precise annotations for notes and multi-pitch, enabling the generation of the necessary target matrices (**Yn**, **Yo**, and **Yp**).
- **Relevance to the Paper**: GuitarSet was used for both training and testing in the paper, making it a relevant benchmark to reproduce their results.

## Dataset Preparation

### Data Download

- GuitarSet data is available on [Zenodo](https://zenodo.org/record/3371780).
- Download the audio files "audio_hex-pickup_debleeded" and the corresponding annotations "annotations".

### Data Preprocessing

The **DataSet** folder contains a notebook titled "Basic Processing of GuitarSet and Target Time-Frequency Matrix Generation." This notebook demonstrates how to preprocess the GuitarSet data and extract useful annotations (notes and multi-pitch). These annotations are used to generate the following binary target matrices:

- **Yn**: Corresponds to note activations (whether a note is active).
- **Yo**: Corresponds to note onsets.
- **Yp**: Corresponds to pitch activations (f0).

### Data Organization

The data is organized into two subfolders:
1. **audio_hex-pickup_debleeded**: Contains the audio files, which serve as inputs to the model.
2. **Matrices**: Contains the time-frequency binary matrices Yn, Yo, and Yp generated for each audio file (the matrices represent the full duration of the audio file). These data will be ready for training the model after segmenting them into 2-second portions.


