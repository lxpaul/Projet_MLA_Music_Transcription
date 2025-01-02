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

The model implementation and training scripts are available in the **"Model Implementation and Training"** folder ................   (à toi de jouer youva si tu veux ajouter des petits trucs ici)
```python
import os
import numpy as np
import librosa
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Activation
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt


class DataGenerator(Sequence):
    def __init__(self, audio_dir, label_dir, batch_size=16, segment_duration=2.0, sr=22050, bins_per_semitone=3, hop_size=0.011, fmin=27.5, midi_min=21, num_semitones=88, n_output_freqs=33, **kwargs):
        """
        Data generator for segmented audio files and labels.
        """
        super().__init__(**kwargs)
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.segment_duration = segment_duration
        self.sr = sr
        self.bins_per_semitone = bins_per_semitone
        self.hop_size = hop_size
        self.fmin = fmin
        self.midi_min = midi_min
        self.num_semitones = num_semitones
        self.bins_per_octave = bins_per_semitone * 12
        self.n_output_freqs = n_output_freqs  # Nombre de bins en sortie après stacking
        self.segment_num_frames = int(segment_duration / hop_size)

        # List of audio files
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

    def __len__(self):
        """Number of batches per epoch."""
        total_segments = sum(self.num_segments_per_audio(audio_file) for audio_file in self.audio_files)
        return int(np.ceil(total_segments / self.batch_size))

    def num_segments_per_audio(self, audio_file):
        """Calculate the number of segments for a specific audio file."""
        audio_path = os.path.join(self.audio_dir, audio_file)
        duration = librosa.get_duration(path=audio_path)
        return int(np.ceil(duration / self.segment_duration))

    def flatten_harmonic_cqt(self, harmonic_cqt):
        """
        Combine harmonic channels into a single channel by flattening the harmonic axis.
        """
        freq, time, harmonics = harmonic_cqt.shape
        flattened_cqt = harmonic_cqt.reshape(freq * harmonics, time)
        return flattened_cqt

    def compute_harmonic_cqt(self, audio, num_frames):
        """
        Compute the CQT and apply harmonic stacking with normalization, including sub-harmonic.
        """
        target_length = int(self.segment_duration * self.sr)
        if len(audio) < target_length:
            padding = np.zeros(target_length - len(audio))
            audio = np.hstack((audio, padding))

        n_bins = self.num_semitones * self.bins_per_semitone
        cqt = librosa.cqt(audio, sr=self.sr, hop_length=int(self.hop_size * self.sr),
                          n_bins=n_bins, bins_per_octave=self.bins_per_octave, fmin=self.fmin)
        cqt = np.abs(cqt)

        harmonics = [0.5] + list(range(1, 8))  # Sous-harmonique et 7 harmoniques
        harmonic_cqt = []
        for h in harmonics:
            shift = int(h * self.bins_per_semitone)
            if h < 1:  # Sous-harmonique
                shifted_cqt = np.roll(cqt, shift=-shift, axis=0)
            else:  # Harmoniques
                shifted_cqt = np.roll(cqt, shift=shift, axis=0)
            harmonic_cqt.append(shifted_cqt)

        harmonic_cqt = np.stack(harmonic_cqt, axis=-1)

        # Réduire à n_output_freqs en tronquant les fréquences
        harmonic_cqt = harmonic_cqt[:self.n_output_freqs, :, :]

        if harmonic_cqt.shape[1] < num_frames:
            padding = np.zeros((harmonic_cqt.shape[0], num_frames - harmonic_cqt.shape[1], harmonic_cqt.shape[2]))
            harmonic_cqt = np.concatenate((harmonic_cqt, padding), axis=1)
        elif harmonic_cqt.shape[1] > num_frames:
            harmonic_cqt = harmonic_cqt[:, :num_frames, :]

        harmonic_cqt = np.log1p(harmonic_cqt)
        harmonic_cqt /= np.max(harmonic_cqt)

        harmonic_cqt_flattened = self.flatten_harmonic_cqt(harmonic_cqt)

        return harmonic_cqt_flattened

    def load_labels(self, label_folder, num_frames):
        """Load binary label matrices (yo, yn, yp) for a specific segment."""
        yo_path = os.path.join(label_folder, "Yo.npy")
        yn_path = os.path.join(label_folder, "Yn.npy")
        yp_path = os.path.join(label_folder, "Yp.npy")

        if not (os.path.exists(yo_path) and os.path.exists(yn_path) and os.path.exists(yp_path)):
            raise FileNotFoundError(f"One or more label files (Yo.npy, Yn.npy, Yp.npy) not found in {label_folder}.")

        yo = np.load(yo_path)[:num_frames]
        yn = np.load(yn_path)[:num_frames]
        yp = np.load(yp_path)[:num_frames]

        return yo, yn, yp

    def __getitem__(self, idx):
        """
        Generate one batch of data.
        """
        X, Y_yo, Y_yn, Y_yp = [], [], [], []

        audio_idx = idx * self.batch_size
        while len(X) < self.batch_size:
            audio_file = self.audio_files[audio_idx % len(self.audio_files)]
            num_segments = self.num_segments_per_audio(audio_file)

            segment_idx = (idx * self.batch_size + len(X)) % num_segments
            audio_path = os.path.join(self.audio_dir, audio_file)
            label_folder = os.path.join(self.label_dir, audio_file.replace("_hex_cln", "").replace(".wav", ""))

            if not os.path.isdir(label_folder):
                raise FileNotFoundError(f"Label folder {label_folder} not found in {self.label_dir}.")

            audio, sr = librosa.load(audio_path, sr=self.sr)

            segment_length = int(self.segment_duration * sr)
            start = segment_idx * segment_length
            end = min((segment_idx + 1) * segment_length, len(audio))
            segment = audio[start:end]

            duration = len(segment) / sr
            if duration < self.segment_duration:
                audio_idx += 1
                continue

            num_frames = int(duration / self.hop_size)

            cqt_harmonic_flattened = self.compute_harmonic_cqt(segment, num_frames)

            yo, yn, yp = self.load_labels(label_folder, num_frames)

            X.append(cqt_harmonic_flattened)
            Y_yo.append(yo)
            Y_yn.append(yn)
            Y_yp.append(yp)

            audio_idx += 1

        X = np.array(X)  # Transformer en tableau NumPy
        X = np.transpose(X, (0, 2, 1))  # Transposer pour correspondre au modèle
        return X, {
            "Yo": np.stack(Y_yo),
            "Yn": np.stack(Y_yn),
            "Yp": np.stack(Y_yp),
        }


# Exemple d'utilisation
if __name__ == "__main__":
    audio_dir = "Dataset/test/audio"
    label_dir = "Dataset/test/matrices"
    batch_size = 16

    generator = DataGenerator(audio_dir, label_dir, batch_size=batch_size, segment_duration=2.0, sr=22050, n_output_freqs=33)

    X, Y = generator[0]

    # Afficher les dimensions de X
    print(f"Shape of X: {X.shape}")  # (batch_size, freq * harmonics, time)
    
    # Afficher toutes les clés et les dimensions des valeurs dans Y
    print("Keys and shapes in Y:")
    for key, value in Y.items():
        print(f"{key}: {value.shape}")


# Définir la fonction de construction du modèle NMP
def build_nmp_model(input_shape):
    """
    Construit le modèle NMP selon la description donnée.
    Args:
        input_shape: Tuple décrivant la forme de l'entrée (hauteur, largeur, canaux).
    Returns:
        Un modèle Keras compilé.
    """
    # Entrée audio transformée (CQT)
    inputs = Input(shape=input_shape, name="CQT_Input")

    # Bloc 1: Extraction de caractéristiques initiales
    x = Conv2D(32, (5, 5), strides=(1, 3), padding="same", name="Conv2D_Block1_1")(inputs)
    x = BatchNormalization(name="BatchNorm_Block1_1")(x)
    x = ReLU(name="ReLU_Block1_1")(x)

    # Bloc 2: Extraction des activations multipitch (Yp)
    yp = Conv2D(16, (5, 5), padding="same", name="Conv2D_Block2_1")(inputs)
    yp = BatchNormalization(name="BatchNorm_Block2_1")(yp)
    yp = ReLU(name="ReLU_Block2_1")(yp)

    yp = Conv2D(8, (3, 39), padding="same", name="Conv2D_Block2_2")(yp)
    yp = BatchNormalization(name="BatchNorm_Block2_2")(yp)
    yp = ReLU(name="ReLU_Block2_2")(yp)

    yp = Conv2D(1, (5, 5), activation="sigmoid", padding="same", name="Yp")(yp)

    # Bloc 3: Extraction des activations de notes (Yn)
    yn = Conv2D(32, (7, 7), strides=(1, 3), padding="same", name="Conv2D_Block3_1")(yp)
    yn = ReLU(name="ReLU_Block3_1")(yn)

    yn = Conv2D(1, (7, 3), activation="sigmoid", padding="same", name="Yn")(yn)

    # Bloc 4: Détection des débuts de notes (Yo)
    yo = Concatenate(name="Concat_Yo")([x, yn])
    yo = Conv2D(1, (3, 3),activation="sigmoid", padding="same", name="Yo")(yo)
    

    # Définir le modèle avec les trois sorties
    model = Model(inputs=inputs, outputs=[yo, yn, yp], name="NMP_Model")

 
    
       # Créer une instance de la perte pondérée pour Yo
    weighted_loss = WeightedBinaryCrossEntropy(positive_weight=0.95, negative_weight=0.05)
    
    # Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "Yo": weighted_loss,
            "Yn": "binary_crossentropy",
            "Yp": "binary_crossentropy",
        },
        loss_weights={"Yo": 1.0, "Yn": 1.0, "Yp": 1.0},
        metrics={"Yo": "accuracy", "Yn": "accuracy", "Yp": "accuracy"}
    )


    return model


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, positive_weight=0.95, negative_weight=0.05, name="weighted_binary_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def call(self, y_true, y_pred):
        """
        Appliquer une binary cross-entropy pondérée.
        """
        # Supprimer la dimension supplémentaire de y_pred si nécessaire
        y_pred = tf.squeeze(y_pred, axis=-1)

        # Calcul manuel de la binary cross-entropy (éviter toute réduction prématurée)
        bce = -(
            y_true * tf.math.log(y_pred + 1e-7) +
            (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
        )

        # Calcul des poids
        weights = y_true * self.positive_weight + (1 - y_true) * self.negative_weight

        # Appliquer les poids
        weighted_bce = weights * bce

        # Calcul final : moyenne sur toutes les dimensions
        return tf.reduce_mean(weighted_bce)

    def get_config(self):
        """
        Configuration pour la sérialisation.
        """
        config = super().get_config()
        config.update({
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstruction de l'objet à partir de la configuration.
        """
        return cls(**config)


# Exemple d'initialisation du modèle
input_shape = (181, 264, 1)  # Exemple de taille d'entrée (CQT avec un canal)
model = build_nmp_model(input_shape)

# Afficher le résumé du modèle
model.summary()
print("Model outputs:", model.output_names)



if __name__ == "__main__":
    # Répertoires des fichiers audio et labels
    audio_dir = "Dataset/train/audio"
    label_dir = "Dataset/train/matrices"
    batch_size = 16

    # Étape 1 : Charger les noms de fichiers audio
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

    # Étape 2 : Diviser les noms de fichiers en 80% train et 20% validation
    train_files, val_files = train_test_split(audio_files, test_size=0.2, random_state=42)
    train_subset = train_files[:100]  # 20 pistes pour l'entraînement
    val_subset = val_files[:30]       # 5 pistes pour la validation


    # Étape 3 : Initialiser les générateurs avec les répertoires et les fichiers sélectionnés
    train_generator = DataGenerator(
        audio_dir=audio_dir,
        label_dir=label_dir,
        batch_size=batch_size,
        segment_duration=2.0,
        sr=22050
    )
    train_generator.audio_files = train_subset  # Limiter aux fichiers train

    val_generator = DataGenerator(
        audio_dir=audio_dir,
        label_dir=label_dir,
        batch_size=batch_size,
        segment_duration=2.0,
        sr=22050
    )
    val_generator.audio_files = val_subset  # Limiter aux fichiers val

    # Étape 4 : Tester les générateurs
    X_train, Y_train = train_generator[0]
    print(f"Train - Shape of X: {X_train.shape}, Shape of Y['yo']: {Y_train['Yo'].shape}")
    print("Keys in Y_batch:", Y_train.keys())  # Doit afficher : dict_keys(['Yo', 'Yp', 'Yn'])


    X_val, Y_val = val_generator[0]
    print(f"Validation - Shape of X: {X_val.shape}, Shape of Y['yo']: {Y_val['Yo'].shape}")

# Calcul automatique des étapes par époque
EPOCHS = 50  # Nombre d'époques
MODEL_PATH = "best_model_weighted_full.keras"

# Calculer le nombre total de segments (échantillons) dans chaque générateur
total_train_samples = sum(train_generator.num_segments_per_audio(audio_file) for audio_file in train_generator.audio_files)
total_val_samples = sum(val_generator.num_segments_per_audio(audio_file) for audio_file in val_generator.audio_files)

STEPS_PER_EPOCH = total_train_samples // train_generator.batch_size
VALIDATION_STEPS = total_val_samples // val_generator.batch_size

print(f"Nombre total d'échantillons pour l'entraînement : {total_train_samples}")
print(f"Nombre total d'échantillons pour la validation : {total_val_samples}")
print(f"STEPS_PER_EPOCH : {STEPS_PER_EPOCH}")
print(f"VALIDATION_STEPS : {VALIDATION_STEPS}")

# Définir les callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

#csv_logger = CSVLogger("training_log_full.csv", append=True)

# Entraîner le modèle
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr],
    verbose=1
)


```

## Using the Trained Model

To use our trained model (`model.h5`), follow these steps:

### Predicting and Visualizing Results

1. Download the trained model (`model.h5`) from the repository.
2. Use example 2-second audio files in the `data_test` folder to test the model.
3. The following Python code loads the model, makes predictions, and visualizes the results:
(youva à toi d'adapter le code si nécessaire! mais je pense il est bon comme ca il suffit juste de créer un script python qui comprend la fonction qui calcul la cqt vu que l'entrée de modèle n'est pas un fichier audio main c'est la cqt après empilment et de l'importer dans ce script)

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
