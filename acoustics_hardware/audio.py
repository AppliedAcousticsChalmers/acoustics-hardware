import sounddevice as sd
from . import core


def get_devices(name=None):
    if name is None:
        return sd.query_devices()
    else:
        return sd.query_devices(name)[name]


class AudioDevice(core.Device):
    def __init__(self, name=None, fs=None, framesize=None):
        if name is None:
            self.name = get_devices()[0]['name']
        else:
            self.name = get_devices(name)

        if fs is None:
            self.fs = sd.query_devices(self.name)['default_samplerate']
        else:
            self.fs = fs

        if framesize is None:
            # TODO: Is this a good default?
            # The sounddevice package allows for variable frames for in/out which is principle is faster, can we make use of this somehow?
            self.framesize = 1024
        else:
            self.framesize = framesize

        self.inputs = []
        self.outputs = []

    def _hardware_run(self):
        raise NotImplementedError

        # Create streams
        if len(self.inputs) and len(self.outputs):
            # in/out stream
        elif len(self.inputs):
            # input stream
        elif len(self.outputs):
            # output stream
        else:
            # Do something??

    def add_input(self, idx):
        if idx not in self.inputs and idx < self.max_inputs:
            self.inputs.append(idx)

    def add_output(self, idx):
        if idx not in self.outputs and idx < self.max_outputs:
            self.outputs.append(idx)

    @property
    def max_inputs(self):
        return sd.query_devices(self.name)['max_input_channels']

    @property
    def max_outputs(self):
        return sd.query_devices(self.name)['max_output_channels']
