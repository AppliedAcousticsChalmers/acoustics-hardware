import queue
import threading
import collections.deque
import logging
from time import sleep
import numpy as np

logger = logging.getLogger(__name__)


class DeviceHandler (threading.Thread):
    # TODO: rename this to something more suitable.
    # It should be possible to make this a sub-process using the multiprocessing module
    # It would require that we change some Events to multiprocessing.Event and some Queues to multiprocessing.Queue.
    # Specifically the Trigger events (how to handle stop for the Device?) and the out queue from the QHandler.
    # The internal queues (device to trigger, trigger to Q) can still be only thread safe, and internaly stop events can also be normal threading.Event
    def __init__(self, device):
        # TODO: Enble shared stop events between multiple devices?
        # Or is it better to handle that with some kind of TriggerHandler
        threading.Thread.__init__(self)
        self.device = device
        self.trigger_handler = TriggerHandler(self.device.Q)
        self.queue_handler = QHandler(self.trigger_handler.Q)
        self._stop_event = threading.Event()
        self.stop = self._stop_event.set

    def run(self):
        logger.info('DeviceHandler started')
        # TODO: Do we want to check if the device is a NIDevice?
        # Will all devices have the possibility to have unscaled reads?
        if self.device.dtype != 'float64':
            self.trigger_handler.trigger_scaling = np.array(self.device.scaling_coeffs())[:, 1].reshape((-1, 1))
        # Start the device, the TriggerHandler, and the QHandler
        self.device.start()
        self.trigger_handler.start()
        self.queue_handler.start()
        # Wait for a stop signal
        self._stop_event.wait()
        logger.info('DeviceHandler stopped')
        # Signal the device to stop
        logger.info('Stopping device')
        self.device.stop()
        logger.info('Joining device')
        self.device.join()
        # Signal the TriggerHandler to stop, wait for it to complete before returning
        logger.info('Stopping TriggerHandler')
        self.trigger_handler.stop()
        logger.info('Joining TriggerHandler')
        self.trigger_handler.join()
        logger.info('Stopping QHandler')
        self.queue_handler.stop()
        logger.info('Joining QHandler')
        self.queue_handler.join()


class TriggerHandler (threading.Thread):
    timeout = 1

    def __init__(self, inQ):
        threading.Thread.__init__(self)
        self._rawQ = inQ
        self.buffer = collections.deque(maxlen=10)
        self.Q = queue.Queue()

        self.trigger_scaling = 1  # Scales the data from the device with this factor. Is mainly used to scale unscaled measurements
        self.running = False
        self.triggers = []  # Holds the Triggers that should be triggered from the data flowing in to the handler.
        self.trigger = threading.Event()

        self.pre_trigger = 0  # Specifies how much before the start trigger we will collect the data. Negative numbers indicate a delay in the aquisition start.
        self.post_trigger = 0  # Specifies how much after the stop trigger we will continue to collect the data. Negative numbers indicate that we stop before something happens, which will give a delay in the flow.

        # self.start_triggers = []  # Holds the triggers that are registred to start the data flow
        # self.stop_triggers = []  # Hold the triggers that will stop the data flow

        self.blocks = 0
        self._stop_event = threading.Event()  # Used to stop the Handler itself, not intended for extenal use.
        self.stop = self._stop_event.set

    def run(self):
        logger.info('TriggerHandler started')
        while not self._stop_event.is_set():
            # Get one block from the input queue
            logger.debug('Fetching block in TriggerHandler')
            try:
                this_block = self._rawQ.get(timeout=self.timeout)  # Note that this will block until there are stuff in the Q. That means that the actual device object needs to put stuff in the Q from a separate thread.
            except queue.Empty:
                # Go directly to checking the stop event again
                # sleep(0.01)
                continue
            logger.debug('Block {} in TriggerHandler'.format(self.blocks))
            self.blocks += 1
            # logger.debug('Block fetched in TriggerHandler loop')
            # Check all trigger conditions for the current block
            for trig in self.triggers:
                trig(this_block * self.trigger_scaling)
            self.buffer.append(this_block)
            # TODO: properply implement pre- and post-triggering both ways
            if self.trigger.is_set():
                # Moves all items from the buffer to the output Q
                while len(self.buffer) > 0:
                        self.Q.put(self.buffer.popleft())


            # TODO: Implement partial blocks depending on where in the block the trigger happens
            # Is this even possible? What would the handler do with the information? The handler does not know which Trigger handler acutually caused the trigger
            # The trigger handler might also be attatched to a different device, which will desync the blocks and the index where the triggering happened is not meaningful anymore.
            # It's only possible to trigger partial blocks exactly if the data stream is triggered by itself.
        # At this point the stop function has been called, finalize then return
        logger.info('TriggerHandler stopped')
        while True:
            try:
                this_block = self._rawQ.get(timeout=self.timeout)
            except queue.Empty:
                break
            for trig in self.triggers:
                trig(this_block)
            self.buffer.append(this_block)

            if self.trigger.is_set():
                # Moves all items from the buffer to the output Q
                while len(self.buffer) > 0:
                        self.Q.put(self.buffer.popleft())
        logger.info('TriggerHandler returning')


class QHandler (threading.Thread):
    timeout = 1  # Global for all QHandlers, specifies how often the stop event will be checked

    def __init__(self, Q):
        threading.thread.__init__(self)
        self._Q = Q
        self.queues = []
        self.blocks = 0

        self._stop_event = threading.Event()  # Used to stop the Handler itself, not intended for extenal use.
        self.stop = self._stop_event.set

    def run(self):
        logger.info('QHandler started')
        while not self._stop_event.is_set():
            # Waits for the trigger to be true
            logger.debug('Waiting for blocks in QHandler')
            try:
                this_block = self._Q.get(timeout=self.timeout)
            except queue.Empty:
                continue
            for Q in self.queues:
                Q.put(this_block)
            # if self.trigger.wait(timeout=self.timeout):
            #     # If we timeout trigger.wait this part will not run, but the stop event will be checked
            #     #while len(self.buffer) > 0:
            #     try:
            #         # There's stuff from the device, move it to the output Qs
            #         this_block = self.buffer.pop()
            #     except IndexError:
            #         sleep(0.01)
            #         continue
            #     logger.debug('Block {} in QHandler'.format(self.blocks))
            #     self.blocks += 1
            #     for Q in self.queues:
            #         # TODO: This will break if the items from the device are immutable, i.e does not have (or need) a copy method
            #         # Is it a problem if the consumers are handed the same object?
            #         Q.put(this_block)
        logger.info('QHandler stopped')
        # At this point the stop function has been called, finalize and return
        while True:
            try:
                this_block = self._Q.get(timeout=self.timeout)
            except queue.Empty:
                break
            for Q in self.queues:
                Q.put(this_block)
        # if self.trigger.wait(timeout=self.timeout):
        #         # If we timeout trigger.wait this part will not run
        #         while len(self.buffer) > 0:
        #             # There's stuff from the device, move it to the output Qs
        #             this_block = self.buffer.pop()
        #             for Q in self.queues:
        #                 # TODO: This will break if the items from the device are immutable, i.e does not have (or need) a copy method
        #                 # Is it a problem if the consumers are handed the same object?
        #                 Q.put(this_block.copy())
        logger.info('QHandler returning')
