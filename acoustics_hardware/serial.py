# from threading import Thread, Event
from serial import Serial
import schunk


def getDevices(name=None):
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


class SerialInstrument:  # (Thread):
    def __init__(self, device=''):
        # Thread.__init__(self)
        self.device = getDevices(device)
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
