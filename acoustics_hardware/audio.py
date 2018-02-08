import sounddevice as sd
from . import core


def get_devices(name=None):
    if name is None:
        return sd.query_devices()
    else:
        pass
        # We need to know if we should return the actual device or just the name of it.


class AudioDevice(core.Device):
    def __init__(self, name='', fs=None, framesize=None):
        pass
        # Get a proper device? Can a sounddevice be pickled?
        # Handle fs and framesize

    def _hardware_setup(self):
        pass
        # Will this be needed?

    def _hardware_reset(self):
        pass
        # Will this be needed?

    def _hardware_run(self):
        raise NotImplementedError

    def _hardware_stop(self):
        raise NotImplementedError
