# import numpy as np
# from threading import Event

from utils import LevelDetector


class RMSTrigger:
    def __init__(self, *, level, channel, action, kind='Above', **kwargs):

        self.level_detector = LevelDetector(channel=channel, **kwargs)
        self.action = action
        self.kind = kind
        self.trigger_level = level

    # TODO: If the loop is not fast enough there are a few options.
    # If we drop the possibility for separate attack and release times, the level detector is
    # just a IIR filter, so we could use scipy.signal.lfilter([1, (1-alpha)], alpha, input_levels)
    # It should also be possible (I think) to write the level detector using Faust, and just drop in the
    # wrapped Faust code here. If I manage to get that working it could also be used for all other crazy
    # filters we might want to use.
    def __call__(self, block):
        # trigger_on = self._event.is_set()
        levels = self.level_detector(block)

        # This will switch the state if the trigger level is passed at least once
        # It should be more robust for transients: If there is a transient that turns on the triggering
        # we do not care if the level dropped afterwards.
        if any(self._kind_sign * levels > self.level * self._kind_sign):
            self.action()

    @property
    def kind(self):
        if self._kind_sign == 1:
            return 'Above'
        else:
            return 'Below'

    @kind.setter
    def kind(self, value):
        if value.lower() == 'above':
            self._kind_sign = 1
        elif value.lower() == 'below':
            self._kind_sign = -1
        else:
            raise ValueError('Kind {} not a valid kind of RMS trigger.'.format(value))
