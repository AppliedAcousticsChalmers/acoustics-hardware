import queue
import sounddevice as sd
from . import core


class AudioDevice(core.Device):
    """Class for interacting with audio interfaces.

    Implementation of the `~core.Device` framework for audio interfaces.
    Built on top of the `sounddevice <http://python-sounddevice.readthedocs.io/>`_ package.

    See Also:
        `acoustics_hardware.core.Device`.
    """
    @staticmethod
    def get_devices(name=None):
        """Check which audio interfaces can be interacted with

        Arguments:
            name (`str`, optional): incomplete name of device or ``None``
        Returns:
            Complete name of device, or list of all devices.
        Todo:
            - The framesize by default set to 1024, is this appropriate?
            - `sounddevice` allows for variable framesizes, can we make use of this somehow?
        """
        if name is None:
            return sd.query_devices()
        else:
            return sd.query_devices(name)['name']

    def __init__(self, name=None, fs=None, framesize=None):
        core.Device.__init__(self)
        if name is None:
            self.name = AudioDevice.get_devices()[0]['name']
        else:
            self.name = AudioDevice.get_devices(name)

        if fs is None:
            self.fs = sd.query_devices(self.name)['default_samplerate']
        else:
            self.fs = fs

        if framesize is None:
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

    @property
    def max_inputs(self):
        return sd.query_devices(self.name)['max_input_channels']

    @property
    def max_outputs(self):
        return sd.query_devices(self.name)['max_output_channels']
