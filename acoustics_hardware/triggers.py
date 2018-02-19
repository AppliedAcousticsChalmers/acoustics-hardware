import numpy as np
import threading
import logging

from . import core, utils

logger = logging.getLogger(__name__)


class RMSTrigger(core.Trigger):
    def __init__(self, level, channel, region='Above', level_detector_args=None, **kwargs):
        core.Trigger.__init__(self, **kwargs)
        self.channel = channel
        # self.level_detector = LevelDetector(channel=channel, fs=fs, **kwargs)
        self.region = region
        self.trigger_level = level
        self.level_detector_args = level_detector_args if level_detector_args is not None else {}

    def setup(self):
        core.Trigger.setup(self)
        self.level_detector = utils.LevelDetector(channel=self.channel, fs=self._device.fs, **self.level_detector_args)

    def test(self, frame):
        # logger.debug('Testing in RMS trigger')
        levels = self.level_detector(frame)
        return any(self._sign * levels > self.trigger_level * self._sign)

    def reset(self):
        core.Trigger.reset(self)
        self.level_detector.reset()

    @property
    def region(self):
        if self._sign == 1:
            return 'Above'
        else:
            return 'Below'

    @region.setter
    def region(self, value):
        if value.lower() == 'above':
            self._sign = 1
        elif value.lower() == 'below':
            self._sign = -1
        else:
            raise ValueError('{} not a valid regoin for RMS trigger.'.format(value))


class PeakTrigger(core.Trigger):
    def __init__(self, level, channel, region='Above', **kwargs):
        core.Trigger.__init__(self, **kwargs)
        self.region = region
        self.trigger_level = level
        self.channel = channel

    def test(self, frame):
        # logger.debug('Testing in Peak triggger')
        levels = np.abs(frame[self.channel])
        return any(self._sign * levels > self.trigger_level * self._sign)

    @property
    def region(self):
        if self._sign == 1:
            return 'Above'
        else:
            return 'Below'

    @region.setter
    def region(self, value):
        if value.lower() == 'above':
            self._sign = 1
        elif value.lower() == 'below':
            self._sign = -1
        else:
            raise ValueError('{} not a valid region for peak trigger.'.format(value))


class DelayedAction:
    def __init__(self, *, action, time):
        self.action = action
        self.time = time
        # self.timer = Timer(interval=time, function=action)

    def __call__(self):
        timer = threading.Timer(interval=self.time, function=self.action)
        timer.start()
        # self.timer.start()
