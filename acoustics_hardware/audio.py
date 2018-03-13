import queue
import sounddevice as sd
from . import core


def get_devices(name=None):
    if name is None:
        return sd.query_devices()
    else:
        return sd.query_devices(name)['name']


class AudioDevice(core.Device):
    def __init__(self, name=None, fs=None, framesize=None):
        core.Device.__init__(self)
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
        # Create streams
        num_output_ch = max(self.outputs, default=-1) + 1
        num_input_ch = max(self.inputs, default=-1) + 1
        self.silent_ch = [ch for ch in range(num_output_ch) if ch not in self.outputs]
        self._hardware_output_timeout = 0.5 * self.framesize / self.fs
        if num_input_ch and num_output_ch:
            # in/out stream
            def callback(indata, outdata, frames, time, status):
                self._input_callback(indata)
                self._output_callback(outdata)
            stream = sd.Stream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=(num_input_ch, num_output_ch))
        elif num_input_ch:
            def callback(indata, frames, time, status):
                self._input_callback(indata)
            stream = sd.InputStream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=num_input_ch)
        elif len(self.outputs):
            def callback(outdata, frames, time, status):
                self._output_callback(outdata)
            stream = sd.OutputStream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=num_output_ch)
        else:
            # Do something??
            pass

        stream.start()
        self._hardware_stop_event.wait()
        stream.stop()

    def _input_callback(self, indata):
        self._hardware_input_Q.put(indata.T[self.inputs].copy())  # Transpose before copy to actually reverse the memory layout

    def _output_callback(self, outdata):
        try:
            outdata[:, self.outputs] = self._hardware_output_Q.get(timeout=self._hardware_output_timeout).T
            outdata[:, self.silent_ch] = 0
        except queue.Empty:
            outdata[:] = 0

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
