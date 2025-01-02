import numpy as np
import librosa

def compute_harmonic_cqt(segment, num_frames=181, sr=22050, bins_per_semitone=3, hop_size=0.011, 
                         fmin=27.5, num_semitones=88, n_output_freqs=33):
    """
    Compute the harmonic Constant-Q Transform (CQT) for a given audio segment.
    Args:
        segment (np.ndarray): The audio signal segment.
        num_frames (int): The number of time frames expected in the output.
        sr (int): Sampling rate of the audio.
        bins_per_semitone (int): Number of frequency bins per semitone.
        hop_size (float): Hop size in seconds for the CQT.
        fmin (float): Minimum frequency for the CQT.
        num_semitones (int): Total number of semitones in the CQT.
        n_output_freqs (int): Number of output frequency bins after harmonic stacking.
    Returns:
        np.ndarray: Flattened harmonic CQT representation of the segment.
    """
    # Define constants
    bins_per_octave = bins_per_semitone * 12
    n_bins = num_semitones * bins_per_semitone
    target_length = int(sr * 2.0)  # Assuming 2 seconds segments

    # Pad the audio if it's shorter than the target length
    if len(segment) < target_length:
        padding = np.zeros(target_length - len(segment))
        segment = np.hstack((segment, padding))

    # Compute the CQT
    cqt = librosa.cqt(segment, sr=sr, hop_length=int(hop_size * sr),
                      n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin)
    cqt = np.abs(cqt)

    # Perform harmonic stacking
    harmonics = [0.5] + list(range(1, 8))  # Sub-harmonic and 7 harmonics
    harmonic_cqt = []
    for h in harmonics:
        shift = int(h * bins_per_semitone)
        if h < 1:  # Sub-harmonic
            shifted_cqt = np.roll(cqt, shift=-shift, axis=0)
        else:  # Harmonics
            shifted_cqt = np.roll(cqt, shift=shift, axis=0)
        harmonic_cqt.append(shifted_cqt)

    harmonic_cqt = np.stack(harmonic_cqt, axis=-1)

    # Truncate or pad the frequency axis
    harmonic_cqt = harmonic_cqt[:n_output_freqs, :, :]
    if harmonic_cqt.shape[1] < num_frames:
        padding = np.zeros((harmonic_cqt.shape[0], num_frames - harmonic_cqt.shape[1], harmonic_cqt.shape[2]))
        harmonic_cqt = np.concatenate((harmonic_cqt, padding), axis=1)
    elif harmonic_cqt.shape[1] > num_frames:
        harmonic_cqt = harmonic_cqt[:, :num_frames, :]

    # Normalize and log scale
    harmonic_cqt = np.log1p(harmonic_cqt)
    harmonic_cqt /= np.max(harmonic_cqt)

    # Flatten the harmonic axis
    freq, time, harmonics = harmonic_cqt.shape
    flattened_cqt = harmonic_cqt.reshape(freq * harmonics, time)

    return flattened_cqt
