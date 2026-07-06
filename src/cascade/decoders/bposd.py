"""BP+OSD decoder wrapper for BB-code baselines.

Builds a `stimbposd.BPOSD` decoder from a Stim circuit's detector error model,
matching the `MWPMDecoder.decode_batch` interface so it plugs into
`eval/decoder_compare.py` without further branching.

For BB codes the DEM contains undecomposable hyperedges, so we pass
``decompose_errors=False`` and rely on stimbposd's
``allow_undecomposed_hyperedges=True`` path.
"""

from __future__ import annotations

import numpy as np
import stim
from stimbposd import BPOSD


class BPOSDDecoder:
    """Decode detection events to logical observable predictions via BP+OSD."""

    def __init__(
        self,
        circuit: stim.Circuit,
        *,
        max_bp_iters: int = 30,
        bp_method: str = "product_sum",
        osd_order: int = 4,
        osd_method: str = "osd_cs",
    ) -> None:
        # osd_order default 4 (was the stimbposd default of 60, which is ~250x
        # slower per shot on BB-72 with no quality gain — empirically osd_order
        # 4 and 10 both give 27.5% block err at p=0.005 over 6 rounds, while
        # osd_order=60 takes ~1 s/shot. See iter-2 §6 for the timing sweep.)
        dem = circuit.detector_error_model(decompose_errors=False)
        self.bposd = BPOSD(
            dem,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            osd_order=osd_order,
            osd_method=osd_method,
        )

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if detection_events.dtype != np.uint8:
            detection_events = detection_events.astype(np.uint8)
        return self.bposd.decode_batch(detection_events).astype(np.uint8)
