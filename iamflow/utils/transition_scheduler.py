
import math
from abc import ABC, abstractmethod
from typing import Optional


class TransitionScheduler(ABC):

    def __init__(self, window_frames: int = 9, delay_frames: int = 3):
        self.window_frames = window_frames
        self.delay_frames = delay_frames
        self.total_frames = delay_frames + window_frames

    @abstractmethod
    def _compute_alpha(self, t: float) -> float:
        pass

    def get_alpha(self, frames_since_switch: int) -> Optional[float]:
        if frames_since_switch >= self.total_frames:
            return None

        if frames_since_switch < self.delay_frames:
            return 0.0

        t = (frames_since_switch - self.delay_frames) / self.window_frames
        t = min(t, 1.0)

        return self._compute_alpha(t)

    def is_complete(self, frames_since_switch: int) -> bool:
        return frames_since_switch >= self.total_frames


class LinearScheduler(TransitionScheduler):

    def _compute_alpha(self, t: float) -> float:
        return t


class CosineScheduler(TransitionScheduler):

    def _compute_alpha(self, t: float) -> float:
        return 0.5 * (1 - math.cos(math.pi * t))


class SigmoidScheduler(TransitionScheduler):

    def __init__(self, window_frames: int = 9, delay_frames: int = 3, steepness: float = 6.0):
        super().__init__(window_frames, delay_frames)
        self.steepness = steepness
        self._low = self._sigmoid(-0.5 * steepness)
        self._high = self._sigmoid(0.5 * steepness)

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _compute_alpha(self, t: float) -> float:
        raw = self._sigmoid(self.steepness * (t - 0.5))
        return (raw - self._low) / (self._high - self._low)


class StepScheduler(TransitionScheduler):

    def __init__(self, window_frames: int = 9, delay_frames: int = 3, frames_per_chunk: int = 3):
        super().__init__(window_frames, delay_frames)
        self.frames_per_chunk = frames_per_chunk
        self.num_steps = max(1, window_frames // frames_per_chunk)

    def _compute_alpha(self, t: float) -> float:
        step = int(t * self.num_steps)
        step = min(step, self.num_steps - 1)
        return (step + 1) / self.num_steps


class AdaptiveScheduler:

    def __init__(self,
                 base_scheduler_type: str = "cosine",
                 min_window: int = 3,
                 max_window: int = 15,
                 delay_frames: int = 3,
                 **scheduler_kwargs):
        self.base_scheduler_type = base_scheduler_type
        self.min_window = min_window
        self.max_window = max_window
        self.delay_frames = delay_frames
        self.scheduler_kwargs = scheduler_kwargs

        self._active_scheduler: TransitionScheduler = create_scheduler(
            scheduler_type=base_scheduler_type,
            window_frames=max_window,
            delay_frames=delay_frames,
            **scheduler_kwargs
        )

    def update_for_switch(self, semantic_distance: float):
        t = max(0.0, min(1.0, semantic_distance))
        window = int(self.min_window + t * (self.max_window - self.min_window))
        window = max(3, (window // 3) * 3)

        self._active_scheduler = create_scheduler(
            scheduler_type=self.base_scheduler_type,
            window_frames=window,
            delay_frames=self.delay_frames,
            **self.scheduler_kwargs
        )

        print(f"[SPT-Adaptive] semantic_distance={t:.3f}, window_frames={window}")

    def get_alpha(self, frames_since_switch: int) -> Optional[float]:
        return self._active_scheduler.get_alpha(frames_since_switch)

    def is_complete(self, frames_since_switch: int) -> bool:
        return self._active_scheduler.is_complete(frames_since_switch)

    @property
    def total_frames(self) -> int:
        return self._active_scheduler.total_frames

    @property
    def window_frames(self) -> int:
        return self._active_scheduler.window_frames


def create_scheduler(
    scheduler_type: str = "cosine",
    window_frames: int = 9,
    delay_frames: int = 3,
    **kwargs
) -> TransitionScheduler:
    schedulers = {
        "linear": LinearScheduler,
        "cosine": CosineScheduler,
        "sigmoid": SigmoidScheduler,
        "step": StepScheduler,
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                         f"Available: {list(schedulers.keys())}")

    cls = schedulers[scheduler_type]

    if scheduler_type == "sigmoid":
        steepness = kwargs.get("steepness", 6.0)
        return cls(window_frames, delay_frames, steepness=steepness)
    elif scheduler_type == "step":
        frames_per_chunk = kwargs.get("frames_per_chunk", 3)
        return cls(window_frames, delay_frames, frames_per_chunk=frames_per_chunk)
    else:
        return cls(window_frames, delay_frames)
