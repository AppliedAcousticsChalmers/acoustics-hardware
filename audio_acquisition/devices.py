from queue import Queue
from threading import Event, Thread
from collections import deque


class DeviceHandler (Thread):
    # TODO: rename this to something more suitable.
    # It should be possible to make this a sub-process using the multiprocessing module
    # It would require that we change some Events to multiprocessing.Event and some Queues to multiprocessing.Queue.
    # Specifically the Trigger events (how to handle stop for the Device?) and the out queue from the QHandler.
    # The internal queues (device to trigger, trigger to Q) can still be only thread safe, and internaly stop events can also be normal threading.Event
    def __init__(self, device):
        # TODO: Enble shared stop events between multiple devices?
        # Or is it better to handle that with some kind of TriggerHandler
        self.device = device
        self.trigger_handler = TriggerHandler(self.device.Q)
        self.queue_handler = QHandler(self.trigger_handler.buffer)
        self._stop_event = Event()
        self.stop = self._stop_event.set

    def run(self):
        # Start the device, the TriggerHandler, and the QHandler
        self.device.start()
        self.trigger_handler.start()
        self.queue_handler.start()
        # Wait for a stop signal
        self._stop_event.wait()
        # Signal the device to stop
        self.device.stop()
        # Signal the TriggerHandler to stop, wait for it to complete before returning
        self.trigger_handler.stop()
        self.trigger_handler.join()
        self.queue_handler.stop()
        self.queue_handler.join()


class TriggerHandler (Thread):
    def __init__(self, inQ):
        self._rawQ = inQ
        self.buffer = deque(maxlen=10)

        self.running = False
        self.triggers = []  # Holds the Triggers that should be triggered from the data flowing in to the handler.

        self.pre_trigger = 0  # Specifies how much before the start trigger we will collect the data. Negative numbers indicate a delay in the aquisition start.
        self.post_trigger = 0  # Specifies how much after the stop trigger we will continue to collect the data. Negative numbers indicate that we stop before something happens, which will give a delay in the flow.

        # self.start_triggers = []  # Holds the triggers that are registred to start the data flow
        # self.stop_triggers = []  # Hold the triggers that will stop the data flow

        self._stop_event = Event()  # Used to stop the Handler itself, not intended for extenal use.
        self.stop = self._stop_event.set

    def run(self):
        while not self._stop_event.is_set():
            # Get one block from the input queue
            this_block = self._rawQ.get()  # Note that this will block until there are stuff in the Q. That means that the actual device object needs to put stuff in the Q from a separate thread.
            # Check all trigger conditions for the current block
            for trig in self.triggers:
                trig(this_block)
            self.buffer.append(this_block)
            # TODO: properply implement pre- and post-triggering both ways

            # TODO: Implement partial blocks depending on where in the block the trigger happens
            # Is this even possible? What would the Q handler do with the information? The Q handler does not know which Trigger handler acutually caused the trigger
            # The trigger handler might also be attatched to a different device, which will desync the blocks and the index where the triggering happened is not meaningful anymore.
            # It's only possible to trigger partial blocks exactly if the data stream is triggered by itself.
        # At this point the stop function has been called, finalize then return
        while self._rawQ.qsize() > 0:
            this_block = self._rawQ.get()
            for trig in self.triggers:
                trig(this_block)
            self.buffer.append(this_block)


class QHandler (Thread):
    timeout = 1  # Global for all QHandlers, specifies how often the stop event will be checked

    def __init__(self, buffer):
        self.buffer = buffer
        self.queues = []
        self.trigger = Event()  # This is the event that should be handed to trigger objects to controll the data flow

        self._stop_event = Event()  # Used to stop the Handler itself, not intended for extenal use.
        self.stop = self._stop_event.set

    def run(self):
        while not self._stop_event.is_set():
            # Waits for the trigger to be true
            if self.trigger.wait(QHandler.timeout):
                # If we timeout trigger.wait this part will not run, but the stop event will be checked
                while len(self.buffer) > 0:
                    # There's stuff from the device, move it to the output Qs
                    this_block = self.buffer.pop()
                    for Q in self.queues:
                        # TODO: This will break if the items from the device are immutable, i.e does not have (or need) a copy method
                        # Is it a problem if the consumers are handed the same object?
                        Q.put(this_block.copy())
        # At this point the stop function has been called, finalize and return
        if self.trigger.wait(QHandler.timeout):
                # If we timeout trigger.wait this part will not run
                while len(self.buffer) > 0:
                    # There's stuff from the device, move it to the output Qs
                    this_block = self.buffer.pop()
                    for Q in self.queues:
                        # TODO: This will break if the items from the device are immutable, i.e does not have (or need) a copy method
                        # Is it a problem if the consumers are handed the same object?
                        Q.put(this_block.copy())
