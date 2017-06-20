import PyDAQmx as daq

def getAvaiableDevices():
	charr = daq.create_string_buffer(256)
	daq.DAQmxGetSysDevNames(charr,256)
	# TODO: Convert to srting, parse for commas, and group in some reasonable manner.
	string = charr.value.decode("utf-8")
	
	print(charr.value)