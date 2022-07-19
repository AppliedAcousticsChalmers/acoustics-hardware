from . import _core
from . import timestamps
import numpy as np
import zarr
import threading
import queue
import os


class _Recorder(_core.SamplerateFollower):
    ...


class InternalRecorder(_Recorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._storage = []

    def process(self, frame):
        self._storage.append(frame)
        return frame

    @property
    def recorded_data(self):
        return np.concatenate(self._storage, axis=-1)


class ZArrRecorder(_Recorder):
    timeout = 1
    def __init__(self, filename=None, mode='w-', chunksize=50000, threaded=False, status_message_interval=None, **kwargs):
        super().__init__(**kwargs)
        if filename is None:
            filename = timestamps.timestamp('filename')
        base, ext = os.path.splitext(filename)
        if ext == '':
            ext = '.zarr'
        self.filename = base + ext
        self.mode = mode
        if self.mode == 'w-' and os.path.exists(self.filename):
            raise FileExistsError(f"File {self.filename} already exists on disk, use overwriting ('w') or appending ('a') mode if desired.")
        self.chunksize = chunksize
        self.threaded = threaded
        self.status_message_interval = status_message_interval
        self._is_ready = False

    def setup(self, **kwargs):
        super().setup(**kwargs)
        if self.threaded:
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self.threaded_writer)
            self._stop_event = threading.Event()
            self._thread.start()
        else:
            self.writer = self.initizlize_writer()
            self.writer.send(None)

    def process(self, frame):
        if not self._is_ready:
            self.setup()
        if self.threaded:
            self._q.put(frame)
        else:
            try:
                self.writer.send(frame)
            except StopIteration:
                pass
        return frame

    def initizlize_writer(self):
        chunksize = self.chunksize
        # shape and chunksize will be ignored if the file exists and mode is append
        storage = zarr.open(self.filename, mode=self.mode, shape=(0, 0), chunks=(1, chunksize), dtype='float64', compressor=None)
        if storage.attrs.setdefault('samplerate', self.samplerate) != self.samplerate:
            raise ValueError(f'Trying to append data with samplerate {self.samplerate} to file with samplerate {storage.attrs["samplerate"]}')
        storage.attrs.setdefault('start_time', timestamps.timestamp())

        if self.status_message_interval not in (False, None):
            if self.status_message_interval is True:
                self.status_message_interval = 1
            status_message_interval = round(self.status_message_interval * self.samplerate)
        else:
            status_message_interval = np.inf
        samps_since_last_status = 0

        write_idx = 0
        frame = (yield)
        channels, framesize = frame.shape
        if storage.shape[0] == 0:
            storage.append(np.zeros(shape=(channels, 0)), axis=0)
        if storage.shape[0] != channels:
            raise ValueError(f'Trying to append {channels} channels to a file with {storage.shape[0]} channels')
        chunksize = storage.chunks[1]  # In case we are appending, we should work with the original chunksize
        buffer = np.zeros(shape=(channels, chunksize))

        while frame is not None:
            framesize = frame.shape[1]

            if framesize < chunksize - write_idx:
                # We can write the entire frame to the buffer
                buffer[:, write_idx:write_idx + framesize] = frame
                write_idx += framesize
                # Get a new frame
                frame = (yield)
            else:
                # The current frame will fill the buffer
                buffer[:, write_idx:] = frame[:, :chunksize - write_idx]
                # Write and reset the buffer, keep the rest of the frame
                storage.append(buffer, axis=1)
                frame = frame[:, chunksize - write_idx:]
                write_idx = 0

                samps_since_last_status += chunksize
                if samps_since_last_status >= status_message_interval:
                    print(f'{storage.shape[1] / self.samplerate:.3f} seconds ({storage.shape[1]} samples) recorded in total', flush=True)
                    samps_since_last_status = 0
        # Write any potential data left in the buffer. This might not be chunk-aligned.
        storage.append(buffer[:, :write_idx], axis=1)



        self._is_ready = False

    def threaded_writer(self):
        writer = self.initizlize_writer()
        writer.send(None)

        while not self._stop_event.is_set():
            try:
                frame = self._q.get(timeout=self.timeout)
            except queue.Empty:
                # Timeout, proceed to check the stop event
                continue
            if frame is None:
                # This signals a stop
                self._stop_event.set()
                continue
            writer.send(frame)
        try:
            writer.send(None)
        except StopIteration:
            pass
