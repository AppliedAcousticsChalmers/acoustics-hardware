import numpy as np
import scipy.signal
from . import _core
import collections


class Gate(_core.SamplerateFollower):
    def __init__(
        self,
        open_trigger=None,
        close_trigger=None,
        pre_trigger=None,
        post_trigger=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.open_trigger = open_trigger
        self.close_trigger = close_trigger
        self.pre_trigger = pre_trigger
        self.post_trigger = post_trigger

    def setup(self, **kwargs):
        super().setup(**kwargs)
        if self.open_trigger is not None:
            self.open_trigger.setup()
        if self.close_trigger is not None:
            self.close_trigger.setup()
        if self.pre_trigger:
            self._pre_trigger_samples = np.math.ceil(self.pre_trigger * self.samplerate)
            self._pre_trigger_buffer = collections.deque()
        if self.post_trigger:
            self._post_trigger_samples = np.math.ceil(self.post_trigger * self.samplerate)
        self.reset()
        #     self._post_trigger_buffer = collections.deque()
        #     self._kept_samples = 0

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.open = True if self.open_trigger is None else False
        self.waiting = True
        self._post_close_buffer = None
        if self.open_trigger is not None:
            self.open_trigger.reset()
        if self.close_trigger is not None:
            self.close_trigger.reset()
        if self.pre_trigger:
            self._pre_trigger_buffer.clear()
            self._buffered_samples = 0
        if self.post_trigger:
            self._remaining_post_trigger_samples = 0

    @property
    def open_trigger(self):
        return self._open_trigger

    @open_trigger.setter
    def open_trigger(self, trigger):
        if trigger is not None:
            self._open_trigger = trigger
            trigger._upstream = self
        else:
            self._open_trigger = None

    @property
    def close_trigger(self):
        return self._close_trigger

    @close_trigger.setter
    def close_trigger(self, trigger):
        if trigger is not None:
            self._close_trigger = trigger
            trigger._upstream = self
        else:
            self._close_trigger = None

    def process(self, frame):
        # The gate can be in the following states, which happen in this order
        # 1) Closed and waiting for the opening trigger
        # 2) Open and waiting for the closing trigger
        # 3) Open and collecting post-trigger data
        # 4) Closed with no intent of opening?
        if self._post_close_buffer is not None:
            frame.frame = np.concatenate((self._post_close_buffer, frame.frame), axis=1)
            self._post_close_buffer = None
        if not self.open and self.waiting and self.open_trigger is not None:
            return self._wait_for_opening_trigger(frame)
        if self.open and self.waiting and self.close_trigger is not None:
            return self._wait_for_closing_trigger(frame)
        if self.open and not self.waiting:
            return self._collect_post_trigger_samples(frame)

    def _wait_for_opening_trigger(self, frame):
        frame = self.open_trigger.process(frame)
        if isinstance(frame, _core.TriggerFrame):
            trig_idx = frame.indices[0]
            frame.frame, post_frame = frame.frame[:, :trig_idx], frame.frame[:, trig_idx:]
        else:
            trig_idx = None

        if self.pre_trigger:
            self._pre_trigger_buffer.append(frame.frame)
            self._buffered_samples += frame.frame.shape[1]
            while self._buffered_samples - self._pre_trigger_buffer[0].shape[1] > self._pre_trigger_samples:
                self._buffered_samples -= self._pre_trigger_buffer.popleft().shape[1]

        if trig_idx is None:
            # Gate is stil closed!
            return

        self.open = True
        self.open_trigger.reset()  # We won't feed data into the open trigger for a while, so it needs to start anew on next call.
        if self.pre_trigger:
            pre_frame = np.concatenate(self._pre_trigger_buffer, axis=1)[:, -self._pre_trigger_samples:]
            self._pre_trigger_buffer.clear()
            self._buffered_samples = 0
        else:
            pre_frame = np.zeros((frame.frame.shape[0], 0))
        # frame = _core.GateOpenFrame(frame.frame[:, ])

        # Run the close trigger on the frame from the opening trigger point, not including the pre-trigger data.
        if self.close_trigger is not None:
            self.waiting = True
            post_frame = self._wait_for_closing_trigger(_core.Frame(post_frame))
        elif self.post_trigger:
            self.waiting = False
            self._remaining_post_trigger_samples = self._post_trigger_samples + 1
            post_frame = self._collect_post_trigger_samples(_core.Frame(post_frame))
        else:
            post_frame = _core.GateCloseFrame(post_frame[:, :1])

        frame = np.concatenate((pre_frame, post_frame.frame), axis=1)
        if isinstance(post_frame, _core.GateCloseFrame):
            return _core.GateOpenCloseFrame(frame)
        return _core.GateOpenFrame(frame)

    def _wait_for_closing_trigger(self, frame):
        frame = self.close_trigger.process(frame)
        if not isinstance(frame, _core.TriggerFrame):
            return frame

        trig_idx = frame.indices[0] + 1
        self.close_trigger.reset()  # We won't feed data into the close trigger for a while, so it needs to start anew on next call.
        pre_frame, post_frame = frame.frame[:, :trig_idx], frame.frame[:, trig_idx:]
        if self.post_trigger:
            self.waiting = False
            self._remaining_post_trigger_samples = self._post_trigger_samples
            post_frame = self._collect_post_trigger_samples(_core.Frame(post_frame))
            post_frame.frame = np.concatenate([pre_frame, post_frame.frame], axis=1)
            return post_frame
        else:
            self.open = False
            self._post_close_buffer = post_frame
            return _core.GateCloseFrame(pre_frame)

    def _collect_post_trigger_samples(self, frame):
        if self._remaining_post_trigger_samples > frame.frame.shape[1]:
            self._remaining_post_trigger_samples -= frame.frame.shape[1]
            return frame

        self.open = False
        self.waiting = True  # TODO: Always reactivate?
        pre_frame, post_frame = frame.frame[:, :self._remaining_post_trigger_samples], frame.frame[:, self._remaining_post_trigger_samples:]
        frame = _core.GateCloseFrame(pre_frame)
        self._remaining_post_trigger_samples = 0
        self._post_close_buffer = post_frame
        return frame


class Trigger(_core.SamplerateFollower):
    def __init__(self, channel=0, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel

    def process(self, frame):
        if isinstance(frame, _core.Frame):
            indices = self.detect(frame.frame[self.channel])
            if indices is not None:
                frame = _core.TriggerFrame(frame.frame, indices)
        return frame

    def detect(self, frame):
        ...


class PeakTrigger(Trigger):
    def __init__(self, threshold=0, edge='rising', **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.edge = edge
        self._prev_val = None

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._prev_val = None

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, val):
        self._threshold = val

    def detect(self, frame):
        if self._prev_val is None:
            self._prev_val = frame[0]
        signs = np.sign(np.insert(frame, 0, self._prev_val) - self._threshold)
        self._prev_val = frame[-1]

        if self.edge == 'crossing':
            test = np.diff(signs)
        elif self.edge == 'rising':
            test = np.diff(signs) > 0
        elif self.edge == 'falling':
            test = np.diff(signs) < 0
        elif self.edge == 'above':
            test = signs > 0
        elif self.edge == 'below':
            test = signs < 0
        if np.any(test):
            # Activation met, we'll stop tracking the levels!
            return np.nonzero(test)[0]


class RMSTrigger(PeakTrigger):
    def __init__(self, threshold, time_constant, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.time_constant = time_constant
        self._state = None

    @property
    def threshold(self):
        return self._threshold ** 0.5

    @threshold.setter
    def threshold(self, val):
        self._threshold = val ** 2

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self._digital_time_coeff = np.exp(-1 / (self.time_constant * self.samplerate))

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._state = None

    def detect(self, frame):
        if self._state is None:
            self._state = scipy.signal.lfilter_zi(
                [1 - self._digital_time_coeff],
                [1, -self._digital_time_coeff],
            ) * frame[0]**2
        levels, self._state = scipy.signal.lfilter(
            [1 - self._digital_time_coeff],
            [1, -self._digital_time_coeff],
            frame**2, zi=self._state
        )

        return super().detect(levels)
