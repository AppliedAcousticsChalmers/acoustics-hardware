# from threading import Thread, Event
from serial import Serial
import schunk

from . import core


def get_devices(name=None):
    from serial.tools.list_ports import comports
    devs = comports()

    # No input given, return the list of all connected devices
    if name is None:
        return devs
    # No name given, return first connected port
    # This is probably never what you want, but we can do nothing about it here
    if len(name) == 0:
        return devs[0].device
    # Try name as a port name
    for dev in devs:
        if dev.device.lower() == name.lower():
            return dev.device
    # Try name as a serial device name
    for dev in devs:
        if dev.description.lower().find(name.lower()) >= 0:
            return dev.device
    # TODO: Any smart way to check for instrument name?
    # This would require us to open a connection for all devs and query for a identifier
    # That's probably dengerous since we might send queries in the wrong format
    # to devices that might respond and do something unspecified


class SerialDevice(core.Device):
    def __init__(self, name):
        core.Device.__init__(self)
        self.name = get_devices(name)
        self.sweeps = 0  # 0 gives steady signal, positive number gives that many sweeps, -1 gives infinite sweeps
        self.shape = 'sine'
        self.frequency = 1e3
        self.frequency_start = 1e3
        self.frequency_stop = 4e3
        self.sweep_time = 1
        self.sweep_spacing = 'log'
        self.amplitude = 1

    sweeps = core.InterProcessAttr('sweeps')
    shape = core.InterProcessAttr('shape')
    frequency = core.InterProcessAttr('frequency')
    frequency_start = core.InterProcessAttr('frequency_start')
    frequency_stop = core.InterProcessAttr('frequency_stop')
    sweep_time = core.InterProcessAttr('sweep_time')
    sweep_spacing = core.InterProcessAttr('sweep_spacing')
    amplitude = core.InterProcessAttr('amplitude')

    def _write(self, *commands):
        for command in commands:
            self.ser.write(bytes(command + '\n', 'UTF-8'))

    def _off(self):
        self._write('apply:DC DEF, DEF, 0')

    def _hardware_reset(self):
        self.ser.close()

    def _sweep_setup(self):
        self._write(
            'frequency:start {}'.format(self.frequency_start),
            'frequency:stop {}'.format(self.frequency_stop),
            'sweep:time {}'.format(self.sweep_time),
            'sweep:spacing {}'.format(self.sweep_spacing)
        )

    def _hardware_run(self):
        self.ser = Serial(port=self.name, timeout=1, dsrdtr=True)
        self._write('system:remote')
        self._off()

        active = False
        if self.sweeps != 0:
            self._sweep_setup()
        while not self._hardware_stop_event.is_set():
            changes = self._update_attrs()
            if self.sweeps == 0:
                # We are using continious output, check if it should be toggled or not
                if active:
                    # Output currently active, check if we should deactivate
                    if not self.output_active.is_set():
                        self._off()
                        active = False
                        # TODO: Will we need time.sleep here? We cannot wait for event to clear,
                        # so the current implementation will only check the event and proceed imediately
                    elif changes:
                        self._write('apply:{} {}, {}'.format(self.shape[:3], self.frequency, self.amplitude))
                else:
                    # Output currently inactive, wait for activation signal
                    if self.output_active.wait(self._hardware_timeout):
                        # TODO: Is this the correct sequencing of the commands?
                        self._write('apply:{} {}, {}'.format(self.shape[:3], self.frequency, self.amplitude))
                        active = True

            else:  # Sweeping mode
                if changes:
                    self._sweep_setup()
                if self.sweeps > 0:
                    # Finite number of sweeps
                    if self.output_active.wait(timeout=self._hardware_timeout):
                        # Do the required sweeps
                        self._write(
                            'func:shape {}'.format(self.shape[:3]),
                            'trigger:source bus',
                            'sweep:state on',
                            'volt {}'.format(self.amplitude))
                        for idx in range(self.sweeps):
                            self._write('*TRG')
                        self._write('*WAI')
                        self._off()
                        self.output_active.clear()
                else:
                    # Infinite number of sweeps
                    if active:
                        # Output currently active, check if we should deactivate
                        if not self.output_active.is_set():
                            self._off()
                            active = False
                            # TODO: Will we need time.sleep here? We cannot wait for event to clear,
                            # so the current implementation will only check the event and proceed imediately
                    else:
                        # Output currently inactive, wait for activation signal
                        if self.output_active.wait(self._hardware_timeout):
                            # TODO: Is this the correct sequencing of the commands?
                            self._write(
                                'func:shape {}'.format(self.shape[:3]),
                                'volt {}'.format(self.amplitude),
                                'sweep:state on')
                            active = True


class SerialInstrument:  # (Thread):
    def __init__(self, device=''):
        # Thread.__init__(self)
        self.device = get_devices(device)
        self.ser = Serial(port=self.device, timeout=1, dsrdtr=True)
        self._write('system:remote')
        self.amplitude = 1

    def __del__(self):
        self.ser.close()

    def _write(self, command):
        '''
        Wrapper for writing to connected device
        '''
        self.ser.write(bytes(command + '\n', 'UTF-8'))

    def sweep_settings(self, *, low, high, time, spacing='log'):
        self._write('frequency:start {}'.format(low))
        self._write('frequency:stop {}'.format(high))
        self._write('sweep:time {}'.format(time))
        self._write('sweep:spacing {}'.format(spacing))

    def single_sweep(self):
        # We need to set the instrument to sine before we set the trigger source, and the amplitude before we start the sweep
        self._write('func:shape sin')
        self._write('trigger:source bus')
        self._write('sweep:state on')
        self._write('volt {}'.format(self.amplitude))
        self._write('*TRG')
        self._write('*WAI')
        self.off()

    def off(self):
        self._write('apply:DC DEF, DEF, 0')

    def sine(self, frequency, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude
        self._write('apply:sin {}, {}'.format(frequency, amplitude))


class VariSphere:
    def __init__(self, az_port='COM1', el_port='COM2'):
        self.az = schunk.Module(schunk.SerialConnection(
            0x0B, Serial, port=az_port, baudrate=9600, timeout=1))
        self.el = schunk.Module(schunk.SerialConnection(
            0x0B, Serial, port=el_port, baudrate=9600, timeout=1))

    def move(self, az, el):
        self.az.move_pos(az)
        self.el.move_pos(el)

    def move_blocking(self, az, el):
        self.move(az, el)
        self.wait()

    def stop(self):
        self.az.stop()
        self.el.stop()

    def wait(self):
        self.az.wait_until_position_reached()
        self.el.wait_until_position_reached()

    def reset(self):
        self.move(0, 0)
