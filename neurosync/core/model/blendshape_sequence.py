from dataclasses import dataclass
from typing import List

@dataclass
class BlendshapeSequence:
    fps: int                 # animation FPS we generated for
    sr: int                  # audio sample rate actually used
    frames: List[List[float]] 