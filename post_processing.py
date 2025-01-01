def post_process_note_events(Yo, Yn,
                             onset_threshold=0.5,
                             activation_threshold=0.3,
                             duration_threshold=120,
                             tolerance_frames=11):
    """
    Post-process posteriorgram matrices (Yo, Yn) to extract note events.

    Args:
        Yo (numpy.ndarray): Onset posteriorgram of shape (88, time_steps).
        Yn (numpy.ndarray): Note activation posteriorgram of shape (88, time_steps).
        onset_threshold (float): Threshold for detecting onsets in Yo.
        activation_threshold (float): Threshold for detecting active notes in Yn.
        duration_threshold (float): Minimum duration for valid notes (in milliseconds).
        tolerance_frames (int): Tolerance for gaps in Yn during note tracking.

    Returns:
        list: List of detected notes as tuples (t_start, t_end, note).
        numpy.ndarray: Binary matrix representing post-processed note events.
    """
    num_pins = Yo.shape[0]  # Number of note bins (88 for notes)
    time_steps = Yo.shape[1]  # Number of time frames
    midi_min = 21  # To converte frame to midi note

    notes = []

    # Convert duration threshold from milliseconds to frames ≈ 120 ms
    min_duration_frames = round((duration_threshold / 11))

    # Step 1: Detect onsets in Yo
    for note in range(num_pins):
        onset_frames = np.where(Yo[note] >= onset_threshold)[0]

        # Sort onset frames in descending order of t^0
        onset_frames = sorted(onset_frames, reverse=True)

        for onset_frame in onset_frames:
            # Track note activation in Yn
            t_start = onset_frame
            t_end = t_start

            # Forward tracking
            for t in range(onset_frame, time_steps):
                if Yn[note, t] >= activation_threshold:
                    t_end = t
                elif t - t_end > tolerance_frames:
                    break

            # Add the detected note if it exceeds the minimum duration
            if t_end - t_start + 1 >= min_duration_frames:
                notes.append((t_start * 0.011, t_end * 0.011, note + midi_min))

                # Zero out used frames in Yn to prevent reuse
                Yn[note, t_start:t_end + 1] = 0

    # Step 2: Create additional notes from Yn
    for note in range(num_pins):
        activation_frames = np.where(Yn[note] > activation_threshold)[0]

        # Sort activation frames in descending order
        activation_frames = sorted(activation_frames, reverse=True)

        for activation_frame in activation_frames:
            t_start = activation_frame
            t_end = activation_frame

            # Backward tracking
            for t in range(activation_frame, -1, -1):
                if Yn[note, t] >= activation_threshold:
                    t_start = t
                elif t_start - t > tolerance_frames:
                    break

            # Forward tracking
            for t in range(activation_frame, time_steps):
                if Yn[note, t] >= activation_threshold:
                    t_end = t
                elif t - t_end > tolerance_frames:
                    break

            # Add the detected note if it exceeds the minimum duration
            if t_end - t_start + 1 >= min_duration_frames:
                notes.append((t_start * 0.011, t_end * 0.011, note + midi_min))

                # Zero out used frames in Yn to prevent reuse
                Yn[note, t_start:t_end + 1] = 0

    # Create binary matrix for post-processed notes
    binary_note_events = np.zeros_like(Yn)
    for t_start, t_end, note in notes:
        start_frame = int(t_start / 0.011)
        end_frame = int(t_end / 0.011)
        binary_note_events[note - midi_min, start_frame:end_frame + 1] = 1

    return notes, binary_note_events