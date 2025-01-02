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

The notebook "Basic Processing of GuitarSet and Target Time-Frequency Matrix Generation" demonstrates how to preprocess the GuitarSet data and extract useful annotations (notes and multi-pitch). These annotations are used to generate the following binary target matrices:

- **Yn**: Corresponds to note activations (whether a note is active).
- **Yo**: Corresponds to note onsets.
- **Yp**: Corresponds to pitch activations (f0).

### Data Organization

We can organize the data into two subfolders:
1. **audio_hex-pickup_debleeded**: Contains the audio files, which serve as inputs to the model. These files will be ready for model training after being segmented into 2-second portions.
2. **matrices**: Contains the time-frequency binary matrices Yn, Yo, and Yp generated for each audio file (the matrices represent the full duration of the audio file). These data will be ready for training the model after segmenting them into 2-second portions.


## Model Implementation and Training

The model implementation and training scripts are available in the **"Model Implementation and Training"** folder which provides everything needed to train the Neural Music Processing (NMP) model. This lightweight, instrument-agnostic model predicts onsets (Yo), note activations (Yn), and multi-pitch estimations (Yp) for polyphonic music transcription.

### Contents:
- **Model Architecture**: Code defining the convolutional network and its multi-output structure.
- **Data Generator**: Scripts for pre-processing audio into harmonic Constant-Q Transforms (CQT) and managing labeled data.
- **Training Scripts**: Includes weighted loss functions, adaptive learning rate adjustments, and checkpoint saving.



## Using the Trained Model

To use our trained model (`model.h5`), follow these steps:

### Predicting and Visualizing Results

1. Download the trained model (`model.h5`) from the repository.
2. Use example 2-second audio files in the `data_test` folder to test the model.
3. The following Python code loads the model, makes predictions, and visualizes the results:

```python
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
# remarque : fais ca dans ton script (segment, num_frames = 181) c mieux je pense pour éviter de calculer dans ce script de démonstartion
from compute_harmonic_cqt import compute_harmonic_cqt   

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load a 2-second audio example, for example, segment_1 of 00_Jazz2-187-F#_solo_hex_cln
audio_path = r"...\data_test\00_Jazz2-187-F#_solo_hex_cln\segments\segment_1.wav"
segment, _ = librosa.load(audio_path, sr=22050)

# To compute the CQT and apply harmonic stacking
cqt_harmonic_flattened = self.compute_harmonic_cqt(segment)

# Make predictions
Yn, Yo, Yp = model.predict(cqt_harmonic_flattened)

# Visualize the results
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.imshow(Yo.T, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
plt.colorbar(label="Value")
plt.title("Yo (Note Onsets)")
plt.xlabel("Time Frames")
plt.ylabel("Frequency Bins")

plt.subplot(1, 3, 2)
plt.imshow(Yn.T, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
plt.colorbar(label="Value")
plt.title("Yn (Note Activations)")
plt.xlabel("Time Frames")
plt.ylabel("Frequency Bins")

plt.subplot(1, 3, 3)
plt.imshow(Yp.T, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
plt.colorbar(label="Value")
plt.title("Yp (Multi-Pitch Estimate)")
plt.xlabel("Time Frames")
plt.ylabel("Frequency Bins")

plt.tight_layout()
plt.show()
```

### Post-Processing
This code is used to process the outputs of models (the posteriorgrams Yo​, Yn​) to create note events (the method used is the one described in the article). For this, we use the post_processing function, which takes the predictions of model as input and returns note events (start time, end time, MIDI note) as well as a binary representation.

```python
from post_processing import post_process_note_events

# Post-process
predicted_note_events, predicted_matrix_note_events = post_process_note_events(Yo.T, Yn.T)

# Plot binary note events after post-processing
plt.figure(figsize=(12, 10))
plt.imshow(predicted_matrix_note_events, aspect='auto', origin='lower', cmap='Greys', vmin=0, vmax=1)
plt.colorbar(label="Value")
plt.title("Note Events (Post-processed)")
plt.xlabel("Time Frames")
plt.ylabel("Frequency Bins")
plt.show()
```

## Comparison with the Original Model
To demonstrate the original Basic Pitch model, a demo website is available to visualize a matrix of note events. To compare our results (the note event matrix visualized in the previous section) with those of the original model:

1. Use the same 2-second example audio file.
2. Upload the audio file to the [Basic Pitch demonstration website](https://basicpitch.spotify.com/).
3. Compare the binary matrix from the original model with those generated by our implementation.


## Evaluation of Metrics
To evaluate the results, we use the metrics as in the article, which include the F-measure (F), the F-measure with no offset (Fno), and note accuracy (Acc) using mir_eval. To do this, simply use the calculate_metrics function as follows:

```python
import numpy as np
import os
import pandas as pd

# Load the ground truth data for a segment, for example, segment_1 of 00_Jazz2-187-F#_solo_hex_cln
segment_folder = r"...\data_test\00_Jazz2-187-F#_solo_hex_cln\segments"
id = "segment_1"
audio_path = os.path.join(segment_folder, f"{id}.wav")   # The 2-second audio
matrix_note_events_path = os.path.join(segment_folder, f"{id}.npy")   # The .npy file containing the note activation matrix
note_events_path = os.path.join(segment_folder, f"{id}.csv")   # The .csv file containing note events in the form (t_start, t_end, note)

# Load the associated files
truth_matrix_note_events = np.load(matrix_note_events_path)
truth_note_events = pd.read_csv(note_events_path)

## Post-process predicted results Yo and Yn (if not already done)
#predicted_note_events, predicted_matrix_note_events = post_process_note_events(Yo, Yn)

# Convert the results into a format compatible with metrics calculation
predicted_note_events_metric = {
    "intervals": [(note[0], note[1]) for note in predicted_note_events],  # Extract t^0 and t^1
    "pitches": [note[2] for note in predicted_note_events]  # Extract f (pitch)
}
truth_note_events_metric = {
    "intervals": [(row["time"], row["time"] + row["duration"]) for _, row in truth_note_events.iterrows()],
    "pitches": truth_note_events["note"].tolist()
}

# Calculating the metrics
metrics = calculate_metrics(truth_note_events_metric, predicted_note_events_metric, truth_matrix_note_events.T, predicted_matrix_note_events)
print(metrics)

```
