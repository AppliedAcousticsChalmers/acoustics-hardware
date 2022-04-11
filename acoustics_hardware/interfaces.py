from . import _core
import sounddevice as sd


class _StreamedInterface(_core.Node):
    def __init__(self, input_channels=None, output_channels=None, samplerate=None, framesize=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samplerate = samplerate
        self.framesize = framesize

        input_channels = input_channels or []
        output_channels = output_channels or []
        try:
            self.input_channels = list(input_channels)
        except TypeError:
            self.input_channels = [input_channels]
        try:
            self.output_channels = list(output_channels)
        except TypeError:
            self.output_channels = [output_channels]


class AudioInterface(_StreamedInterface):
    @classmethod
    def list_interfaces(cls):
        """Check which audio interfaces can be interacted with.

        This is just a wrapper around `sounddevice.query_devices`.

        Returns
        -------
        interfaces : list
            A list of all devices.
        """
        return sd.query_devices()

    def __init__(self, name=None, input_name=None, output_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_name = input_name or name
        output_name = output_name or name
        input_name = sd.default.device[0] if input_name is None else input_name
        output_name = sd.default.device[1] if output_name is None else output_name

        device_list = sd.query_devices()
        self._input_device = sd.query_devices(input_name)
        self._input_device['index'] = device_list.index(self._input_device)
        self._output_device = sd.query_devices(output_name)
        self._output_device['index'] = device_list.index(self._output_device)

    def run(self):
        outputs = self.output_channels
        inputs = self.input_channels
        num_input_ch = max(inputs, default=-1) + 1
        num_output_ch = max(outputs, default=-1) + 1
        silent_ch = [ch for ch in range(num_output_ch) if ch not in outputs]

        def process_input(frame):
            # Take the frame read from hardware and push it downstream in the pipeline.
            self._output.input(frame.T[inputs])

        def process_output(buffer):
            # Get a frame from upstream in the pipeline and write it to the buffer.
            try:
                buffer[:, outputs] = self._input.output(buffer.shape[0]).T
                buffer[:, silent_ch] = 0
            except _core.PipelineStop:
                raise sd.CallbackStop()

        if num_input_ch and num_output_ch:
            def callback(indata, outdata, frames, time, status):
                process_input(indata)
                process_output(outdata)
            self._stream = sd.Stream(
                samplerate=self.samplerate,
                blocksize=self.framesize,
                device=(self._input_device['index'], self._output_device['index']),
                channels=(num_input_ch, num_output_ch),
                callback=callback,
            )
        elif num_input_ch:
            def callback(indata, frames, time, status):
                process_input(indata)
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                blocksize=self.framesize,
                device=self._input_device['name'],
                channels=num_input_ch,
                callback=callback,
            )
        elif num_output_ch:
            def callback(outdata, frames, time, status):
                process_output(outdata)
            self._stream = sd.OutputStream(
                samplerate=self.samplerate,
                blocksize=self.framesize,
                device=self._output_device['name'],
                channels=num_output_ch,
                callback=callback,
            )

        self._stream.start()

    def stop(self):
        self._stream.stop()

    def process(self, frame=None, framesize=None):
        outputs = [ch + 1 for ch in self.output_channels]
        inputs = [ch + 1 for ch in self.input_channels]

        if len(inputs) and len(outputs):
            return sd.playrec(
                data=frame.T,
                samplerate=self.samplerate,
                device=(self._input_device['index'], self._output_device['index']),
                input_mapping=inputs,
                output_mapping=outputs,
                blocking=True,
            )
        elif len(inputs):
            return sd.rec(
                frames=framesize,
                samplerate=self.samplerate,
                device=self._input_device['index'],
                mapping=inputs,
                blocking=True,
            )
        elif len(outputs):
            return sd.play(
                data=frame.T,
                samplerate=self.samplerate,
                device=self._output_device['index'],
                mapping=outputs,
                blocking=True,
            )

    @property
    def max_inputs(self):
        return self._input_device['max_input_channels']

    @property
    def max_outputs(self):
        return self._output_device['max_output_channels']