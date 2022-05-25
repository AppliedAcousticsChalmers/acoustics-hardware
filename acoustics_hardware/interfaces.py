from . import _core
import numpy as np
import sounddevice as sd
import time
import functools


def _channel_collection(channel_class, **default_channel_kwargs):
    @functools.wraps(channel_class, updated=())
    class ChannelCollection:
        default_channel_settings = default_channel_kwargs
        def __init__(self, channels=None):
            self.channels = []
            if channels is None:
                return

            if isinstance(channels, int):
                # Channels given as number of channels to use
                channels = range(channels)
            if isinstance(channels, dict):
                # Channels given as a single dict defining one channel
                channels = [channels]

            try:
                # Channels given as a single pair of (int, dict) for a single channel
                ch, configs = channels
            except (ValueError, TypeError):
                pass
            else:
                if isinstance(ch, int) and isinstance(configs, dict):
                    # This extra check is needed in case channels is given as e.g. [0, 1]
                    # which defines two channels, not one.
                    channels = [channels]

            for channel in channels:
                try:
                    # This channel is a single dict defining the channel
                    self.append(**channel)
                except TypeError:
                    pass
                else:
                    continue

                try:
                    # Channel is an (int, dict) pair
                    ch, configs = channel
                except (ValueError, TypeError):
                    pass
                else:
                    if isinstance(ch, int) and isinstance(configs, dict):
                        self.append(channel=ch, **configs)
                        continue

                if isinstance(channel, int):
                    # This channel i s just defined by its index, with no settings
                    self.append(channel)
                    continue
                raise ValueError(f'Cannot add channel with definition {channel}')


        def append(self, channel, **kwargs):
            self.channels.append(channel_class(channel, **self.default_channel_settings | kwargs))

        def __getitem__(self, key):
            return self.channels[key]

        def __len__(self):
            return len(self.channels)

        def __repr__(self):
            return repr(self.channels)

    return ChannelCollection


class SimpleChannel:
    """A single channel without settings.

    Parameters
    -----------
    channel : int
        The index of the channel, zero based.
    label : string, default none
        An optional label to assign to the channel.
    """

    def __init__(self, channel, label=None):
        self.channel = channel
        self.label = label

    def __repr__(self):
        label = '' if self.label is None else f", '{self.label}'"
        return f'{self.__class__.__name__}({self.channel}{label})'


class _StreamedInterface(_core.SamplerateDecider):
    _input_channels = _channel_collection(SimpleChannel)
    _output_channels = _channel_collection(SimpleChannel)

    def __init__(self, input_channels=None, output_channels=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = self._input_channels(input_channels)
        self.output_channels = self._output_channels(output_channels)


class AudioInterface(_StreamedInterface):
    @classmethod
    def list_interfaces(cls):
        """Check which audio interfaces can be interacted with.

        This is just a wrapper around `sounddevice.query_devices`.

        Returns
        -------
        interfaces : list
            A list of all interfaces found.
        """
        return sd.query_devices()

    def __init__(self, name=None, input_name=None, output_name=None, framesize=None, **kwargs):
        super().__init__(**kwargs)
        self.framesize = framesize
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
        outputs = [ch.channel for ch in self.output_channels]
        inputs = [ch.channel for ch in self.input_channels]
        num_input_ch = max(inputs, default=-1) + 1
        num_output_ch = max(outputs, default=-1) + 1
        if num_output_ch == 1 and self.max_outputs > 1:
            # PortAudio by default duplicates mono channels over all channels.
            # This is not what the rest of this package is doing.
            num_output_ch = 2
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
                buffer[:] = 0
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
        outputs = [ch.channel + 1 for ch in self.output_channels]
        inputs = [ch.channel + 1 for ch in self.input_channels]

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


            self._input_task.register_every_n_samples_acquired_into_buffer_event(framesize_downstream, self._read_callback)

        if len(self.output_channels):
            self._output_task.timing.cfg_samp_clk_timing(
                rate=samplerate_upstream,
                samps_per_chan=self.buffer_n_frames * framesize_upstream,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
            )

            self._write = nidaqmx.stream_writers.AnalogMultiChannelWriter(self._output_task.out_stream).write_many_sample
            self._output_task.out_stream.regen_mode = nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
            # self._write(np.zeros((self._output_task.out_stream.num_chans, framesize_upstream)))
            self._write_countdown = None
            self._samples_written = 0
            self._output_frames = []
            self._write_callback(None, None, self.buffer_n_frames * framesize_downstream, None)
class DummyInterface(_StreamedInterface):
    def __init__(self, framesize=None, **kwargs):
        super().__init__(**kwargs)
        self.framesize = framesize

    def run(self):
        import threading
        self._running = threading.Event()
        self._thread = threading.Thread(target=self._passthrough_target)
        self._thread.start()

    def _passthrough_target(self):
        self._running.set()
        frames = 0
        while self._running.is_set() and frames < 200:
            frames += 1
            try:
                frame = self._input.output(self.framesize)
            except _core.PipelineStop:
                self._running.clear()
                break
            self._output.input(frame)

    def stop(self):
        self._running.clear()
