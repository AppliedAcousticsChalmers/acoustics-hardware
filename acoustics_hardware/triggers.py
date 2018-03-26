import numpy as np
import threading
import logging

from . import core, utils

logger = logging.getLogger(__name__)


class RMSTrigger(core.Trigger):
    """RMS level trigger.

    Triggers actions based on a detected root-mean-square level.

    Arguments:
        level (`float`): The level at which to trigger.
        channel (`int`): The index of the channel on which to trigger.
        region (``'Above'`` or ``'Below'``, optional): Defines if the triggering
            happens when the detected level rises above or falls below the set
            level, default ``'Above'``.
        level_detector_args (`dict`, optional): Passed as keyword arguments to
            the internal `~.utils.LevelDetector`.
        **kwargs: Extra keyword arguments passed to `.core.Trigger`.
    """
    def __init__(self, level, channel, region='Above', level_detector_args=None, **kwargs):
        core.Trigger.__init__(self, **kwargs)
        self.channel = channel
        # self.level_detector = LevelDetector(channel=channel, fs=fs, **kwargs)
        self.region = region
        self.trigger_level = level
        self.level_detector_args = level_detector_args if level_detector_args is not None else {}

    def setup(self):
        core.Trigger.setup(self)
        self.level_detector = utils.LevelDetector(channel=self.channel, device=self.device, **self.level_detector_args)

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
            raise ValueError('{} not a valid region for RMS trigger.'.format(value))


class PeakTrigger(core.Trigger):
    """Peak level trigger.

    Triggers actions based on detected peak level.

    Arguments:
        level (`float`): The level at which to trigger.
        channel (`int`): The index of the channel on which to trigger.
        region (``'Above'`` or ``'Below'``, optional): Defines if the triggering
            happens when the detected level rises above or falls below the set
            level, default ``'Above'``.
        **kwargs: Extra keyword arguments passed to `.core.Trigger`.
    """
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
    """Delays an action.

    When called, an instance of this class will excecute a specified action
    after a set delay. This can be useful to create timed measurements or
    pauses in a longer sequence.

    Arguments:
        action (callable): Any callable action. This can be a callable class,
            a user defined funciton, or a method of another class.
            If several actions are required, create a lambda that calls all
            actions when called.
        time (`float`): The delay time, in seconds.
    """
    def __init__(self, action, time):
        self.action = action
        self.time = time
        # self.timer = Timer(interval=time, function=action)

    def __call__(self):
        timer = threading.Timer(interval=self.time, function=self.action)
        timer.start()
        # self.timer.start()
