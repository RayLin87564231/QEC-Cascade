"""PyMatching-based MWPM decoder for surface codes.

Builds a `pymatching.Matching` object directly from a Stim circuit's
detector error model. The DEM is the standard QEC-baseline interface.
"""

from __future__ import annotations

import numpy as np
import pymatching
import stim


class MWPMDecoder:
    """Decode detection events to logical observable predictions via MWPM."""

    def __init__(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(decompose_errors=True)
        self.matching = pymatching.Matching.from_detector_error_model(dem)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Vectorised decode.

        Args:
            detection_events: ``(N, num_detectors)`` uint8/bool/int array.

        Returns:
            ``(N, num_observables)`` predicted logical flips, uint8.
        """
        if detection_events.dtype != np.uint8:
            detection_events = detection_events.astype(np.uint8)
        return self.matching.decode_batch(detection_events).astype(np.uint8)
