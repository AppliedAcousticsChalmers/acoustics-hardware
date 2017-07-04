import PyDAQmx as daq
from queue import Queue
from threading import Event
import numpy as np
from time import sleep

def getAvaiableDevices():
	charr = daq.create_string_buffer(256)
	daq.DAQmxGetSysDevNames(charr,256)
	# TODO: Convert to srting, parse for commas, and group in some reasonable manner.
	string = charr.value.decode("utf-8")
	return [s.strip() for s in string.split(',')]
	#print(charr.value)

class NIdevice:
	def __init__(self, device):		
		self.task = CallbackTask(device)
		
		# TODO: Figure out how to get the values for these
		self.inputChannels = list(range(4)) # This will hold a list of input channels in the end
		self.outputChannels = list(range(0)) # This will hold a list of output channels in the end
		for idx in self.inputChannels:
			self.task.registerAnalogueInput(idx)
		for idx in self.outputChannels:
			self.task.registerAnalogueOutput(idx)
		self.task.blocksize = 10000 # This is the only time this should be changed manually.
		self.fs = int(1e6)


	@property
	def fs(self):
		return self.task.fs
	@fs.setter
	def fs(self, fs):
		# TODO: Fetch the allowed fs
		allowedFs = [int(1e3), int(1e4), int(1e5), int(1e6)]
		if not int(fs) in allowedFs:
			raise ValueError("Not an allowed sampling frequency for this device")
		self.task.setTiming(fs,self.blocksize)

	@property
	def blocksize(self):
		return self.task.blocksize
	@blocksize.setter
	def blocksize(self,blocksize):
		self.task.setTiming(self.fs,blocksize)

	def start(self):
		self.running = True
		self.task.start()
	def stop(self):
		self.task.stop()
		
		self.indata = np.empty((self.blocksize*self.task.blocks,self.task.inchannels))
		self.task.UnregisterEveryNSamplesEvent()
		#b = daq.bool32()
		#b.value = 0
		#self.waited = 0
		#while not b.value:
		#	self.task.GetTaskComplete(b)
		#	self.waited += 1
		#	sleep(2)
		#print(self.waited)
		popped = 0

		#while self.task.inQ.not_empty:
		#while self.task.inQ.unfinished_tasks>0:
			#data = self.task.inQ.get()
			#self.indata[popped*self.blocksize:(popped+1)*self.blocksize,:] = data
			#self.task.inQ.task_done()
		#	popped += 1

		self.running=False





	def __registerAnalogueInputChannel(self, idx):
		#TODO: Get this value from the device instead
		assert 0<=idx<4
		#TODO: Get the values for this?
		#TODO: Make these values configurable?
		rangeMin = -10
		rangeMax = 10
		self.task.CreateAIVoltageChan(self.device+"/ai{}".format(idx), 'Ch{}In'.format(idx), daq.DAQmx_Val_Default, rangeMin, rangeMax, daq.DAQmx_Val_Volts, None)

	def __registerAnalogueOutputChannel(self, idx):
		#TODO: Get this value from the device instead
		assert 0<=idx<0
		#TODO: Get the values for this?
		#TODO: Make these values configurable?
		rangeMin = -10
		rangeMax = 10
		self.task.CreateAOVoltageChan(self.device+"/ao{}".format(idx), 'Ch{}Out'.format(idx), daq.DAQmx_Val_Default, rangeMin, rangeMax, daq.DAQmx_Val_Volts, None)

	


class CallbackTask(daq.Task):
	def __init__(self,device):
		daq.Task.__init__(self)
		if not device in getAvaiableDevices():
			raise ValueError("No such device connected")
		self.device = device
		self.AutoRegisterDoneEvent(0)
		self.SetReadAutoStart(1)
		self.stopFlag = False
		self.doneEvent = Event()
	def setTiming(self,fs,blocksize):
		self.CfgSampClkTiming("", fs, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, blocksize)
		self.fs = fs
		self.blocksize = blocksize
	def registerAnalogueInput(self, idx):
		if not hasattr(self, "inQ"):
			#self.inQ = Queue()
			self.inQ = []
			self.inchannels=0
		assert 0<=idx<4 # TODO:Get value from device
		# TODO: Get allowed values
		# TODO: Make these values configurable
		rangeMin = -10
		rangeMax = 10
		self.CreateAIVoltageChan(self.device+"/ai{}".format(idx), "Ch{}In".format(idx), daq.DAQmx_Val_Default, rangeMin, rangeMax, daq.DAQmx_Val_Volts, None)
		self.inchannels += 1
	def DoneCallback(self, status):
		print("Status",status.value)
		return 0
	def inputCallback(self):
		#b = daq.bool32()
		#self.GetTaskComplete(b)
		#assert not b.value
		read = daq.int32()
		self.ReadAnalogF64(self.blocksize, 10.0, daq.DAQmx_Val_GroupByScanNumber, self.indata, self.indata.size, daq.byref(read), None)
		self.inQ.extend(self.indata.tolist())
		#self.inQ.put(self.indata.copy())
		self.blocks += 1
		#print(self.blocks)
		#if self.stopFlag:
		#	self.StopTask()
		#	self.doneEvent.set()
		return 0 # This should return a status int
	def start(self):
		self.stopFlag = False
		self.doneEvent.clear()
		if hasattr(self,"inQ") and hasattr(self,"outQ"):
			raise NotImplementedError()
		elif hasattr(self,"outQ"):
			raise NotImplementedError()
		elif hasattr(self,"inQ"):
			self.indata = np.zeros((self.blocksize, self.inchannels))
			self.AutoRegisterEveryNSamplesEvent(daq.DAQmx_Val_Acquired_Into_Buffer, self.blocksize, 0, 'inputCallback')
			self.blocks = 0
			self.StartTask()
			#assert False
	def stop(self):
		#self.stopFlag = True
		#self.doneEvent.wait()
		self.StopTask()
			


#### TODO: This class show the callback functionality. 
# It seems as if the definition of the callback function needs to be inside a Tack class.
# This forces some relocation of data. The NIdevice class needs to own a CallbackTask instead of a Task.
# The callback task needs to be able to reach the queue, the blocksize, and possible other things.
# PROP: Move blocksize, device, fs, to the CallbackTask and access them from the NIdevice using @property as if they were members of the NIdiveice instance.
# This should allow changing of the settings on the NIdevice object without accessing the Task object directly, while still mantaining proper access from the Callback function.
#class CallbackTask(Task):
#    def __init__(self):
#        Task.__init__(self)
#        self.data = zeros(1000)
#        self.a = []
#        self.CreateAIVoltageChan("Dev1/ai0","",DAQmx_Val_RSE,-10.0,10.0,DAQmx_Val_Volts,None)
#        self.CfgSampClkTiming("",10000.0,DAQmx_Val_Rising,DAQmx_Val_ContSamps,1000)
#        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,1000,0)
#        self.AutoRegisterDoneEvent(0)
#    def EveryNCallback(self):
#        read = int32()
#        self.ReadAnalogF64(1000,10.0,DAQmx_Val_GroupByScanNumber,self.data,1000,byref(read),None)
#        self.a.extend(self.data.tolist())
#        print self.data[0]
#        return 0 # The function should return an integer
#    def DoneCallback(self, status):
#        print "Status",status.value
#        return 0 # The function should return an integer