from queue import Queue
from threading import Event, Thread
from collections import deque


class Device (Thread):
    def __init__(self, device):
        # TODO: Enble shared stop events between multiple devices?
        # Or is it better to handle that with some kind of TriggerHandler
        self.device = device
        self._rawQ = device.Q
        self.Q = Queue()
        self.trigger = TriggerHandler(self._rawQ, self.Q)
        self._stop_event = Event()
        self.stop = self._stop_event.set

    def run(self):
        # Start the device actual
        self.device.start()
        self.trigger.start()
        # Start the TriggerHandler
        while not self._stop_event.is_set():
            # This will not wait, but will break the loop when the stop function is called
            raise NotImplementedError('The Device object is not yet ready for use.')
        # Signal the device to stop
        self.device.stop()
        # Signal the TriggerHandler to stop, wait for it to complete before returning
        self.trigger.stop()


class TriggerHandler:
    def __init__(self, inQ, outQ):
        self._rawQ = inQ
        self.Q = outQ
        self._pre_trig_deque = deque(maxlen=0)

        self.running = False
        self.triggers = []  # Holds the Triggers that should be triggered from the data flowing in to the handler.

        self.start_triggers = []  # Holds the triggers that are registred to start the data flow
        self.pre_trigger = 0  # Specifies how much before the start trigger we will collect the data. Negative numbers indicate a delay in the aquisition start.

        self.stop_triggers = []  # Hold the triggers that will stop the data flow
        self.post_trigger = 0  # Specifies how much after the stop trigger we will continue to collect the data. Negative numbers indicate that we stop before something happens, which will give a delay in the flow.

        self._stop_event = Event()  # Used to stop the Handler itself, not intended for extenal use.
        self.stop = self._stop_event.set

    def start(self):
        while not self._stop_event.is_set():
            # Get one block from the input queue
            this_block = self._rawQ.get()  # Note that this will block until there are stuff in the Q. That means that the actual device object needs to put stuff in the Q from a separate thread.
            # Check all trigger conditions for the current block
            for trig in self.triggers:
                trig(this_block)

            # Check all registred conditions
            if self.running and any([trig.is_active() for trig in self.stop_triggers]):
                self.running = False
            elif any([trig.is_active() for trig in self.start_triggers]):  # This does not allow start and stop in the same block.
                self.running = True
                # We just started, move everything from the deque to the Q
                while len(self._pre_trig_deque) > 0:
                    self.Q.put(self._pre_trig_deque.pop())

            if self.running:
                self.Q.put(this_block)
            else:
                self._pre_trig_deque.append(this_block)
