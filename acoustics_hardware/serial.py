# from threading import Thread, Event
from serial import Serial
import schunk


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


class SerialGenerator:  # (Thread):
    def __init__(self, name, frequency=1e3, amplitude=1, output_unit='rms', shape='sine',
                 sweeps=0, sweep_time=1, sweep_start=100, sweep_stop=1e3, sweep_spacing='log'):
        # Thread.__init__(self)
        self.device = get_devices(name)
        self.ser = Serial(port=self.device, timeout=1, dsrdtr=True)
        self._write('system:remote', 'output:load inf')
        self.output_unit = output_unit
        self.amplitude = amplitude
        self.frequency = frequency
        self.shape = shape
        self.sweeps = sweeps
        self.sweep_time = sweep_time
        self.sweep_start = sweep_start
        self.sweep_stop = sweep_stop
        self.sweep_spacing = sweep_spacing
        self._prev_sweep_settings = {
            'time': None,
            'start': None,
            'stop': None,
            'spacing': None
        }

    def __del__(self):
        self.ser.close()

    def _write(self, *commands):
        '''
        Wrapper for writing to connected device
        '''
        for command in commands:
            self.ser.write(bytes(command + '\n', 'UTF-8'))

    def _read(self):
        return self.ser.readline().decode()[:-2]

    def check_errors(self):
        errors = []
        while True:
            self._write('system:error?')
            error = self._read()
            if error == '+0,"No error"':
                break
            else:
                errors.append(error)
        if len(errors) == 0:
            return False
        else:
            return errors

    def _sweep_setup(self):
        if not self.sweep_start == self._prev_sweep_settings['start']:
            self._prev_sweep_settings['start'] = self.sweep_start
            self._write('frequency:start {}'.format(self.sweep_start))
        if not self.sweep_stop == self._prev_sweep_settings['stop']:
            self._prev_sweep_settings['stop'] = self.sweep_stop
            self._write('frequency:stop {}'.format(self.sweep_stop))
        if not self.sweep_time == self._prev_sweep_settings['time']:
            self._prev_sweep_settings['time'] = self.sweep_time
            self._write('sweep:time {}'.format(self.sweep_time))
        if not self.sweep_spacing == self._prev_sweep_settings['spacing']:
            self._prev_sweep_settings['spacing'] = self.sweep_spacing
            self._write('sweep:spacing {}'.format(self.sweep_spacing))

    def off(self):
        self._write('apply:DC DEF, DEF, 0')

    def on(self):
        if self.sweeps == 0:
            self._write('apply:{} {}, {}'.format(self.shape, self.frequency, self.amplitude))
        elif self.sweeps < 0:
            self._sweep_setup()
            self._write(
                'func:shape {}'.format(self.shape),
                'volt {}'.format(self.amplitude),
                'sweep:state on')
        else:
            self._sweep_setup()
            self._write(
                'func:shape {}'.format(self.shape),
                'trigger:source bus',
                'sweep:state on',
                'volt {}'.format(self.amplitude),
                *self.sweeps * ('*TRG',),
                '*WAI')
            self.off()

    @property
    def output_unit(self):
        self._write('voltage:unit?')
        return self._read()

    @output_unit.setter
    def output_unit(self, value):
        if value.lower() == 'rms':
            self._write('voltage:unit VRMS')
        elif value.lower() == 'peak':
            self._write('voltage:unit VPP')

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if value.lower()[:3] == 'sin':
            self._shape = 'sinusoid'
        elif value.lower()[:3] == 'squ':
            self._shape = 'square'
        elif value.lower()[:3] == 'tri':
            self._shape = 'triangle'
        elif value.lower()[:3] == 'noi':
            self._shape = 'noise'
        elif value.lower()[:3] == 'dc':
            self._shape = 'dc'


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
