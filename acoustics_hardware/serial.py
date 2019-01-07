import numpy as np
from serial import Serial, serial_for_url
import schunk


class SerialDevice:
    @staticmethod
    def get_devices(name=None):
        from serial.tools.list_ports import comports
        devs = comports()
        # No input given, return the list of all connected devices
        if name is None:
            return [dev.device for dev in devs]
        # No name given, return first connected port
        # This is probably never what you want, but we can do nothing about it here
        if len(name) == 0:
            return devs[0].device
        # Try name as a port name
        for dev in devs:
            if dev.device.lower().find(name.lower()) >= 0:
                return dev.device
        # Try name as a serial device name
        for dev in devs:
            if dev.description.lower().find(name.lower()) >= 0:
                return dev.device

    def __init__(self, name=None, **kwargs):
        if name is not None:
            kwargs.setdefault('port', SerialDevice.get_devices(name))
        kwargs.setdefault('timeout', 1)
        self.ser = Serial(**kwargs)

    def __del__(self):
        self.ser.close()

    def _write(self, *commands):
        '''
        Wrapper for writing to connected device
        '''
        for command in commands:
            self.ser.write(bytes(command + '\n', 'UTF-8'))

    def _read(self):
        return self.ser.readline().decode().split('\n')[0]


class GCodeMotion(SerialDevice):
    def __init__(self, name, size=None, **kwargs):
        kwargs.setdefault('baudrate', 115200)
        super().__init__(name=name, **kwargs)
        self.offsets = np.array([0, 0, 0])
        self.size = np.array([np.inf, np.inf, np.inf]) if size is None else np.asarray(size)
        self._write('G21', 'G90')  # Sets units to mm, absolute coordinates
        self.ser.reset_input_buffer()
        self._write('G0')  # The first command is sometimes ignored

    def move(self, x=None, y=None, z=None, speed=None):
        x0, y0, z0 = self.offsets
        xmax, ymax, zmax = self.size
        x_comm = '' if x is None else ' X{:3.3f}'.format(xmax if x + x0 > xmax else 0 if x + x0 < 0 else x + x0)
        y_comm = '' if y is None else ' Y{:3.3f}'.format(ymax if y + y0 > ymax else 0 if y + y0 < 0 else y + y0)
        z_comm = '' if z is None else ' Z{:3.3f}'.format(zmax if z + z0 > zmax else 0 if z + z0 < 0 else z + z0)
        spd_comm = '' if speed is None else ' F{:3.3f}'.format(speed * 60)
        command = 'G0' + x_comm + y_comm + z_comm + spd_comm
        self._write(command)

    def home_axis(self, *args):
        command = 'G28'
        for ax in args:
            command = command + ' ' + ax.upper()
        self._write(command)

    def set_origin(self):
        self.offsets = self.absolute_position

    @property
    def absolute_position(self):
        self._write('M114')
        response = self._read()
        self._read()
        xidx = response.find('X:')
        yidx = response.find('Y:')
        zidx = response.find('Z:')
        eidx = response.find('E:')
        x = float(response[xidx + 2:yidx])
        y = float(response[yidx + 2:zidx])
        z = float(response[zidx + 2:eidx])
        return np.array([x, y, z])

    @property
    def position(self):
        return self.absolute_position - self.offsets

    def halt(self):
        self._write('M112')

    def _read(self):
        while True:
            line = super()._read()
            if line != 'ok':
                return line


class SerialGenerator(SerialDevice):
    def __init__(self, name, frequency=1e3, amplitude=1, output_unit='rms', shape='sine',
                 sweeps=0, sweep_time=1, sweep_start=100, sweep_stop=1e3, sweep_spacing='log', **kwargs):
        kwargs.setdefault('dsrdtr', True)
        super().__init__(name=name, **kwargs)
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
    """Class for controlling a VariSphere

    Supports two modes, comport access or ip access. Comport usage means that
    there is a comport listed in the system which is connected to the motors.
    Ip usage means that there is a TCP port on the system which is connected
    to the motors.

    Arguments:
        az_port (`str`): The port for the azimuth motor. Specify `None` or `False`
            to not use this motor. Default 4001.
        el_port (`str`): The port for the elevation motor. Specify `None` or `False`
            to not use this motor. Default 4002.
        ip (`str`): Ip adress of the ethernet-to-serial interface. Specify `None`
            or `False` to use comports mode. Default `192.168.127.120`.

    """
    def __init__(self, az_port='4001', el_port='4002', ip='192.168.127.120'):
        self.az = None
        self.el = None
        if ip:
            if az_port:
                self.az = schunk.Module(schunk.SerialConnection(
                    0x0B, serial_for_url, url='socket://' + ip + ':' + str(az_port),
                    baudrate=9600, timeout=1))
            if el_port:
                self.el = schunk.Module(schunk.SerialConnection(
                    0x0B, serial_for_url, url='socket://' + ip + ':' + str(el_port),
                    baudrate=9600, timeout=1))
        else:
            if az_port:
                self.az = schunk.Module(schunk.SerialConnection(
                    0x0B, Serial, port=az_port, baudrate=9600, timeout=1))
            if el_port:
                self.el = schunk.Module(schunk.SerialConnection(
                    0x0B, Serial, port=el_port, baudrate=9600, timeout=1))

    def move(self, az=None, el=None):
        if az is not None:
            self.az.move_pos(az)
        if el is not None:
            self.el.move_pos(el)

    def move_blocking(self, az=None, el=None):
        self.move(az, el)
        self.wait()

    def stop(self):
        if self.az is not None:
            self.az.stop()
        if self.el is not None:
            self.el.stop()

    def wait(self):
        if self.az is not None:
            self.az.wait_until_position_reached()
        if self.el is not None:
            self.el.wait_until_position_reached()

    def reset(self):
        if self.az is not None:
            self.az.move_pos(0)
        if self.el is not None:
            self.el.move_pos(0)
