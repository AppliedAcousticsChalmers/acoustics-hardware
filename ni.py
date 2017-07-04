import nidaqmx as daq

def getAvailableDevices():
	system = daq.system.System.local()
	return [dev.name for dev in system.devices]