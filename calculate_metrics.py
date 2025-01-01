import mir_eval
import numpy as np


def calculate_metrics(ground_truth_notes, predicted_notes, ground_truth_frames, predicted_frames):
    """
    Calculates the F, Fno, and Acc metrics to evaluate music transcription.

    Args:
        ground_truth_notes (dict): Ground truth notes with keys "intervals" and "pitches".
                                   - "intervals": List of intervals [(t_start, t_end), ...].
                                   - "pitches": List of corresponding pitches [f1, f2, ...].
        predicted_notes (dict): Predicted notes with the same keys "intervals" and "pitches".
        ground_truth_frames (numpy.ndarray): Binary matrix (num_pitches, num_frames) for the ground truth.
        predicted_frames (numpy.ndarray): Binary matrix (num_pitches, num_frames) for predictions.

    Returns:
        dict: Contains the F, Fno, and Acc metrics as a dictionary.
    """
    # Convert lists to numpy.ndarray
    ref_intervals = np.array(ground_truth_notes["intervals"])
    ref_pitches = np.array(ground_truth_notes["pitches"])
    est_intervals = np.array(predicted_notes["intervals"])
    est_pitches = np.array(predicted_notes["pitches"])

    # Parameters for tolerance
    onset_tolerance = 0.05  # 50 ms
    offset_ratio = 0.2  # 20% of the note's duration

    # Compute transcription metrics using mir_eval
    scores = mir_eval.transcription.evaluate(
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        est_intervals=est_intervals,
        est_pitches=est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio
    )

    # Extract F (full F-measure with offset)
    precision = scores["Precision"]
    recall = scores["Recall"]
    f_measure = scores["F-measure"]

    # Calculate Fno (F-measure without offset)
    scores_fno = mir_eval.transcription.evaluate(
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        est_intervals=est_intervals,
        est_pitches=est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=None  # Ignore offsets
    )

    # Extract Fno (note without offset)
    f_measure_no_offset = scores_fno["F-measure_no_offset"]

    # Manual calculation of Acc (frame accuracy)
    def frame_accuracy(gt_frames, pred_frames):
        # Check the dimensions of the matrices
        if gt_frames.shape != pred_frames.shape:
            raise ValueError("The dimensions of the frame matrices do not match.")
        # Compare frame-by-frame activations
        correct_frames = np.sum(gt_frames == pred_frames)
        total_frames = gt_frames.size
        return correct_frames / total_frames

    frame_accuracy_value = frame_accuracy(ground_truth_frames, predicted_frames)

    # Results as a dictionary
    results = {
        "F": {
            "Precision": precision,
            "Recall": recall,
            "F-measure": f_measure
        },
        "Fno": f_measure_no_offset,
        "Acc": frame_accuracy_value
    }

    return results
