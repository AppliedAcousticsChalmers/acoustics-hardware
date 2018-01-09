import queue
import threading
import multiprocessing
import collections
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Device:
    """
    Top level abstract Device class used for inheritance to implement specific hardware.
    Required methods to implement:
    - `run_hardware`

    Optional methods to implement
    - `stop_hardware`: The default implementation is `self.stop_hardware_event.set()`.
      Either make `run_hardware` work with that, or implement a new stop method.
    """
    _trigger_timeout = 1
    _q_timeout = 1

    def __init__(self):
        self.trigger = multiprocessing.Event()

        self._hardware_Q = queue.Queue()
        self._hardware_pause_event = multiprocessing.Event()
        self._hardware_stop_event = threading.Event()

        self.__triggers = []
        self.__Qs = []
        self.__triggered_q = queue.Queue()

        self.__process_stop_event = multiprocessing.Event()
        self.__trigger_stop_event = threading.Event()
        self.__q_stop_event = threading.Event()
        self.__process = multiprocessing.Process()

    def start(self):
        self.__process = multiprocessing.Process(target=self.__process_target)
        self.__process.start()

    def stop(self):
        self.__process_stop_event.set()
        self.__process.join()

    def pause(self):
        self._hardware_pause()

    def _hardware_run(self):
        '''
        This is the primary method in which data processing should be implemented.
        '''
        raise NotImplementedError('Required method `_hardware_run` not implemented in {}'.format(self.__class__.__name__))

    def _hardware_stop(self):
        '''
        This is the primary method used for stopping the hardware.
        It is reccomended to do this using Events inside the _hardware_run method.
        The default implementation is
            self._hardware_pause_event.set()
            self._hardware_stop_event.set()
        '''
        self._hardware_pause_event.set()
        self._hardware_stop_event.set()

    def _hardware_pause(self):
        '''
        This is the primary method used for pausing the hardware. Pausing the hardware will not allow
        changes in the intra-process setup, e.g. triggers or queues, but can be used to pause the data
        flow while waiting for something else.
        Prefferably, paused hardware should not actually do any reads or writes at all.
        '''
        self._hardware_pause_event.set()

    def get_output_Q(self):
        if self.__process.is_alive():
            # TODO: Custom warning class
            raise UserWarning('It is not possible to register new Qs while the device is running. Stop the device and perform all setup before starting.')
        else:
            Q = multiprocessing.Queue()
            self.__Qs.append(Q)
            return Q

    def remove_output_Q(self, Q):
        if self.__process.is_alive():
            # TODO: Custom warning class
            raise UserWarning('It is not possible to remove Qs while the device is running. Stop the device and perform all setup before starting.')
        else:
            # TODO: What should happen if the Q is not in the list?
            self.__Qs.remove(Q)

    def register_trigger(self, trigger):
        if self.__process.is_alive():
            # TODO: Custom warning class
            raise UserWarning('It is not possible to register new triggers while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__triggers.append(trigger)

    def remove_trigger(self, trigger):
        if self.__process.is_alive():
            # TODO: Custom warning class
            raise UserWarning('It is not possible to remove triggers while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__triggers.remove(trigger)

    def __process_target(self):
        # Start hardware in separate thread
        # Manage triggers in separate thread
        # Manage Qs in separate thread
        hardware_thread = threading.Thread(target=self._hardware_run)
        trigger_thread = threading.Thread(target=self.__trigger_target)
        q_thread = threading.Thread(target=self.__q_target)

        hardware_thread.start()
        trigger_thread.start()
        q_thread.start()

        self.__process_stop_event.wait()

        self._hardware_stop()
        hardware_thread.join()
        self.__trigger_stop_event.set()
        trigger_thread.join()
        self.__q_stop_event.set()
        q_thread.join()

    def __trigger_target(self):
        # TODO: Get buffer size depending on pre-trigger value
        data_buffer = collections.deque(maxlen=10)

        # TODO: Get trigger scaling if needed
        trigger_scaling = 1

        while not self.__trigger_stop_event.is_set():
            # Wait for a block, if none has arrived within the set timeout, go back and check stop condition
            try:
                this_block = self._hardware_Q.get(timeout=self._trigger_timeout)
            except queue.Empty:
                continue
            # Execute all triggering conditions
            for trig in self.__triggers:
                trig(this_block * trigger_scaling)
            # Move the block to the buffer
            data_buffer.append(this_block)
            # If the trigger is active, move everything from the data buffer to the triggered Q
            if self.trigger.is_set():
                while len(data_buffer) > 0:
                    self.__triggered_q.put(data_buffer.popleft())

        # The hardware should have stopped by now, analyze all remaining blocks.
        while True:
            try:
                this_block = self._hardware_Q.get(timeout=self._trigger_timeout)
            except queue.Empty:
                break
            for trig in self.__triggers:
                trig(this_block * trigger_scaling)
            data_buffer.append(this_block)
            if self.trigger.is_set():
                while len(data_buffer) > 0:
                    self.__triggered_q.put(data_buffer.popleft())

    def __q_target(self):
        while not self.__q_stop_event.is_set():
            # Wait for a block, if none has arrived within the set timeout, go back and check stop condition
            try:
                this_block = self.__triggered_q.get(timeout=self._q_timeout)
            except queue.Empty:
                continue
            for Q in self.__Qs:
                # Copy the block to all output Qs
                Q.put(this_block)

        # The triggering should have stopped by now, move the remaining blocks.
        while True:
            try:
                this_block = self.__triggered_q.get(timeout=self._q_timeout)
            except queue.Empty:
                break
            for Q in self.__Qs:
                Q.put(this_block)


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
