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


## Model Implementation and Training

The model implementation and training scripts are available in the **Model Implementation and Training** folder. Detailed instructions in the provided notebooks will guide you through the training process using the prepared dataset.

## Using the Trained Model

To use our trained model (`model.h5`), follow these steps:

### Predicting and Visualizing Results

1. Download the trained model (`model.h5`) from the repository.
2. Use example 2-second audio files in the `examples` folder to test the model.
3. The following Python code loads the model, makes predictions, and visualizes the results:

```python
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load a 2-second audio example
audio_path = "examples/example_audio.wav"
y, sr = librosa.load(audio_path, sr=22050, duration=2.0)

# Preprocess the audio (convert to Constant-Q Transform)
cqt = librosa.cqt(y, sr=sr, n_bins=84, bins_per_octave=12)
cqt = np.abs(cqt).T  # Transpose for compatibility

# Make predictions
predictions = model.predict(np.expand_dims(cqt, axis=0))

# Visualize results
Yn, Yo, Yp = predictions
plt.figure(figsize=(15, 5))
plt.subplot(3, 1, 1)
plt.title("Yn (Note Activations)")
plt.imshow(Yn[0], aspect='auto', origin='lower')
plt.colorbar()
plt.subplot(3, 1, 2)
plt.title("Yo (Note Onsets)")
plt.imshow(Yo[0], aspect='auto', origin='lower')
plt.colorbar()
plt.subplot(3, 1, 3)
plt.title("Yp (Multi-Pitch Estimate)")
plt.imshow(Yp[0], aspect='auto', origin='lower')
plt.colorbar()
plt.tight_layout()
plt.show()
```

### Post-Processing

To generate note events (start time, end time, pitch) from the predictions, use the `post_processing.py` script:
```python
from post_processing import create_note_events

# Post-process predictions
note_events = create_note_events(Yn[0], Yo[0], Yp[0])

# Print the resulting note events
for note in note_events:
    print(f"Pitch: {note['pitch']}, Start: {note['start_time']}s, End: {note['end_time']}s")
```

## Comparison with the Original Model

To compare results with the original model:

1. Use the same 2-second example audio files.
2. Upload the audio files to the [Basic Pitch demonstration website](https://basicpitch.github.io/).
3. Export the note event results and compare the matrices from the original model with those generated by this implementation.
