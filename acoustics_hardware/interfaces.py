from . import _core, signal_tools
import numpy as np
import sounddevice as sd
import time
import functools

try:
    import nidaqmx
    import nidaqmx.stream_readers
    import nidaqmx.stream_writers
except ImportError:
    pass


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
            self._downstream.push(frame.T[inputs])

        def process_output(buffer):
            # Get a frame from upstream in the pipeline and write it to the buffer.
            try:
                buffer[:, outputs] = self._upstream.request(buffer.shape[0]).T
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


class NationalInstrumentsConfigurableChannel(SimpleChannel):
    """A configurable National instruments channel

    Parameters
    ----------
    channel : int
        The index of the channel, zero based.
    label : str, default None
        Optional label to assign the channel
    voltage_range : numeric
        The range in which to configure the channel.
        Give a two values (min, max) or a single amplitude value.
    terminal_config : str, default None
        How to configure the terminals for the channel. Leave as None to use device default configuration.
        Should be one of
        - Differential: for a differential input
        - Single ended: for a single ended measurement
    """
    def __init__(self, channel, label=None, voltage_range=None, terminal_config=None):
        super().__init__(channel=channel, label=label)
        self.voltage_range = voltage_range
        self.terminal_config = terminal_config

    def config(self, interface):
        configs = {}
        if self.voltage_range is not None:
            try:
                min_val, max_val = self.voltage_range
            except TypeError:
                min_val = -self.voltage_range
                max_val = self.voltage_range
            configs['min_val'] = min_val
            configs['max_val'] = max_val
        if self.terminal_config is not None:
            if 'diff' in self.terminal_config.lower():
                configs['terminal_config'] = nidaqmx.constants.TerminalConfiguration.BAL_DIFF
            elif 'single' in self.terminal_config.lower():
                configs['terminal_config'] = nidaqmx.constants.TerminalConfiguration.RSE
            else:
                raise ValueError(f'Unknown terminal configuration {self.terminal_config}')
        return configs

    def __repr__(self):
        s = super().__repr__()[:-1]
        voltage_range = '' if self.voltage_range is None else f', voltage_range={self.voltage_range}'
        terminal_config = '' if self.terminal_config is None else f', terminal_config={self.terminal_config}'
        return s + voltage_range + terminal_config + ')'


class NationalInstrumentsInputChannel(NationalInstrumentsConfigurableChannel):
    def config(self, interface):
        return super().config(interface) | dict(physical_channel=f'{interface.name}/ai{self.channel}')


class NationalInstrumentsOutputChannel(NationalInstrumentsConfigurableChannel):
    def config(self, interface):
        return super().config(interface) | dict(physical_channel=f'{interface.name}/ao{self.channel}')


class NationalInstrumentsDaqmx(_StreamedInterface):
    @classmethod
    def list_interfaces(cls):
        """Check what National Instruments hardware is available.

        Returns
        -------
        interfaces : list
            A list of all devices.
        """
        try:
            system = nidaqmx.system.System.local()
        except NameError as e:
            if e.args[0] == "name 'nidaqmx' is not defined":
                raise ModuleNotFoundError("Windows-only module 'nidaqmx' is not installed")
            else:
                raise e
        name_list = [dev.name for dev in system.devices]
        return name_list


    def __init__(self, name=None, framesize=None, buffer_n_frames=25, **kwargs):
        self.name = name or self.list_interfaces()[0]
        self._device = nidaqmx.system.Device(self.name)
        self._input_channels = _channel_collection(
            NationalInstrumentsInputChannel,
            terminal_config='single ended',
            voltage_range=(min(self._device.ai_voltage_rngs), max(self._device.ai_voltage_rngs))
        )
        self._output_channels = _channel_collection(
            NationalInstrumentsOutputChannel,
            voltage_range=(min(self._device.ao_voltage_rngs), max(self._device.ao_voltage_rngs))
        )
        super().__init__(**kwargs)
        self._is_ready = False  # This class needs a proper setup!
        self.framesize = framesize
        self.buffer_n_frames = buffer_n_frames

        if self.samplerate is None:
            num_in = self.max_inputs
            num_out = self.max_outputs
            if num_in and num_out:
                fs_in = self._device.ai_max_multi_chan_rate
                fs_out = self._device.ao_max_rate
                self.samplerate = (fs_out, fs_in)
            elif num_in:
                self.samplerate = self._device.ai_max_multi_chan_rate
            elif num_out:
                self.samplerate = self._device.ao_max_rate
            else:
                raise ValueError(f'National Instruments interface {self.name} seems to have neither inputs nor outputs')

    def reset(self, **kwargs):
        super().reset(**kwargs)
        try:
            self._device.reset_device()
        except (AttributeError, nidaqmx.DaqError):
            pass

    @property
    def max_inputs(self):
        return len(self._device.ai_physical_chans)

    @property
    def max_outputs(self):
        return len(self._device.ao_physical_chans)

    def setup(self, finite_samples=False, **kwargs):
        super().setup(**kwargs)

        if len(self.input_channels):
            self._input_task = nidaqmx.Task()
            for ch in self.input_channels:
                self._input_task.ai_channels.add_ai_voltage_chan(**ch.config(self))

        if len(self.output_channels):
            self._output_task = nidaqmx.Task()
            for ch in self.output_channels:
                self._output_task.ao_channels.add_ao_voltage_chan(**ch.config(self))

        try:
            samplerate_upstream, samplerate_downstream = self.samplerate
        except TypeError:
            samplerate_upstream = samplerate_downstream = self.samplerate

        if finite_samples is False:
            try:
                framesize_upstream, framesize_downstream = self.framesize
            except TypeError:
                framesize_upstream = framesize_downstream = self.framesize
            samps_per_chan_read = self.buffer_n_frames * framesize_downstream
            samps_per_chan_write = self.buffer_n_frames * framesize_upstream
            sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
        else:
            try:
                samps_per_chan_write, samps_per_chan_read = finite_samples
            except TypeError:
                samps_per_chan_write = samps_per_chan_read = finite_samples
            sample_mode = nidaqmx.constants.AcquisitionType.FINITE

        if len(self.input_channels):
            self._input_task.timing.cfg_samp_clk_timing(
                rate=samplerate_downstream,
                samps_per_chan=samps_per_chan_read,
                sample_mode=sample_mode
            )
            if finite_samples is False:
                self._databuffer = np.empty((self._input_task.in_stream.num_chans, framesize_downstream))
                self._read = nidaqmx.stream_readers.AnalogMultiChannelReader(self._input_task.in_stream).read_many_sample
                self._samples_read = 0
                self._input_task.register_every_n_samples_acquired_into_buffer_event(framesize_downstream, self._read_callback)

        if len(self.output_channels):
            self._output_task.timing.cfg_samp_clk_timing(
                rate=samplerate_upstream,
                samps_per_chan=samps_per_chan_write,
                sample_mode=sample_mode
            )

            if finite_samples is False:
                self._write = nidaqmx.stream_writers.AnalogMultiChannelWriter(self._output_task.out_stream).write_many_sample
                self._output_task.out_stream.regen_mode = nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION
                self._samples_written = 0
                self._write_callback(None, None, self.buffer_n_frames * framesize_downstream, None)
                self._output_task.register_every_n_samples_transferred_from_buffer_event(framesize_upstream, self._write_callback)

        if len(self.output_channels) and len(self.input_channels):
            self._input_task.triggers.start_trigger.cfg_dig_edge_start_trig(f'/{self.name}/ao/StartTrigger')


    def run(self):
        if not self._is_ready:
            self.setup(pipeline=True)

        if len(self.input_channels):
            self._input_task.start()
        if len(self.output_channels):
            self._output_task.start()

    def stop(self):
        self._is_ready = False  # The tasks are single use!

        if len(self.output_channels):
            try:
                self._output_task.stop()
                self._output_task.wait_until_done(timeout=10)
                self._output_task.close()
            except nidaqmx.DaqError:
                pass

        if len(self.input_channels):
            while self._samples_read < self._samples_written:
                time.sleep(self.framesize / self.samplerate)
            try:
                self._input_task.stop()
                self._input_task.wait_until_done(timeout=10)
                self._input_task.close()
            except nidaqmx.DaqError:
                pass

    def _read_callback(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        try:
            self._read(self._databuffer, number_of_samples)
        except nidaqmx.DaqError:
            self.stop()
            raise
        self._samples_read += number_of_samples
        self._downstream.push(self._databuffer.copy())
        return 0

    def _write_callback(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        try:
            frame = self._upstream.request(number_of_samples).squeeze()
        except _core.PipelineStop:
            frame = np.zeros((len(self.output_channels), number_of_samples))
            self._write(frame)
            while self._output_task.out_stream.total_samp_per_chan_generated < self._samples_written:
                time.sleep(self._output_task.out_stream.sleep_time)
            self.stop()
            return 0

        self._write(frame)
        self._samples_written += number_of_samples
        return 0

    def process(self, frame=None, framesize=None):
        if frame is not None:
            padded_frame = signal_tools.pad_signals(frame, post_pad=1)
        if framesize is None and len(self.input_channels):
            try:
                samplerate_upstream, samplerate_downstream = self.samplerate
            except TypeError:
                samplerate_upstream = samplerate_downstream = self.samplerate
            framesize = np.math.ceil(padded_frame.shape[-1] * samplerate_downstream / samplerate_upstream)
        self.setup(finite_samples=(padded_frame.shape[-1], framesize), pipeline=True)

        if len(self.input_channels):
            self._input_task.start()

        if len(self.output_channels):
            timeout = frame.shape[-1] / samplerate_upstream * 1.2 + 5
            self._output_task.write(padded_frame.squeeze())
            self._output_task.start()
            self._output_task.wait_until_done(timeout=timeout)
            self._output_task.close()

        if len(self.input_channels):
            timeout = framesize / samplerate_downstream * 1.2 + 5
            read_frame = self._input_task.read(framesize, timeout=timeout)
            read_frame = np.asarray(read_frame)
            self._input_task.close()
            return read_frame[..., 1:]


class NationalInstrumentsMultimoduleInputChannel(NationalInstrumentsConfigurableChannel):
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module

    def __repr__(self):
        s = super().__repr__()
        pre, post = s.split('(')
        return f'{pre}(module={self.module}, {post}'

    def config(self, interface):
        return interface.input_modules[self.module].ai_physical_chans[self.channel].name


class NationalInstrumentsMultimoduleOutputChannel(NationalInstrumentsMultimoduleInputChannel):
    def config(self, interface):
        return interface.output_modules[self.module].ao_physical_chans[self.channel].name


class CompactDaqmxMultimodule(NationalInstrumentsDaqmx):
    @classmethod
    def get_chassis(cls, name=None):
        devices = cls.list_interfaces()
        chassis = [dev for dev in devices if 'cDAQ' in dev and 'Mod' not in dev]
        if name is None:
            return chassis
        if name not in chassis:
            name = [dev for dev in chassis if str(name) in dev][0]
        return nidaqmx.system.Device(name)

    @classmethod
    def get_modules(cls, chassis):
        try:
            chassis = chassis.name
        except AttributeError:
            pass
        chassis = cls.get_chassis(chassis)
        devs = cls.list_interfaces()
        return [nidaqmx.system.Device(dev) for dev in devs if chassis in dev and 'Mod' in dev]

    @classmethod
    def reset_chassis_modules(cls, chassis):
        for module in cls.get_modules(chassis):
            nidaqmx.system.Device(module).reset_device()

    _input_channels = _channel_collection(NationalInstrumentsMultimoduleInputChannel)
    _output_channels = _channel_collection(NationalInstrumentsMultimoduleOutputChannel)

    def __init__(self, name=None, **kwargs):
        if name is None:
            name = self.list_devices()[0]
        self.chassis = self.get_chassis(name)
        self.modules = self.get_modules(self.chassis)

        # We need to set a default name for the NIDAQ class to initialize properly
        if len(self.input_modules) > 0:
            self.name = self.input_modules[0]
        elif len(self.output_modules) > 0:
            self.name = self.output_modules[0]
        super().__init__(**kwargs)
        del self.name

        self.input_modules = [module for module in self.modules if len(module.ai_physical_chans) > 0]
        self.output_modules = [module for module in self.modules if len(module.ao_physical_chans) > 0]

    @property
    def max_inputs(self):
        inputs = 0
        for module in self.input_modules:
            inputs += len(module.ai_physical_chans)
        return inputs

    @property
    def max_outputs(self):
        outputs = 0
        for module in self.output_modules:
            outputs += len(module.ao_physical_chans)
        return outputs

    def setup(self, **kwargs):
        super().setup(**kwargs)
        if len(self.input_channels) and len(self.output_channels):
            self._output_task.timing.samp_clk_timebase_src = self._input_task.timing.samp_clk_timebase_src
            self._output_task.timing.samp_clk_timebase_rate = self._input_task.timing.samp_clk_timebase_rate

    def reset(self, **kwargs):
        super().reset(**kwargs)
        for module in self.input_modules:
            module.reset_device()
        for module in self.output_modules:
            module.reset_device()


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
                frame = self._upstream.request(self.framesize)
            except _core.PipelineStop:
                self._running.clear()
                break
            self._downstream.push(frame)

    def stop(self):
        self._running.clear()
