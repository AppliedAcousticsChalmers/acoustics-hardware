import numpy as np

from . import _core


class _Recorder(_core.SamplerateFollower):
    ...


class InternalRecorder(_Recorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._storage = []

    def process(self, frame):
        self._storage.append(frame)
        return frame

    @property
    def recorded_data(self):
        return np.concatenate(self._storage, axis=-1)
