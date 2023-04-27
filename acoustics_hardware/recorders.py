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
        self._storage = [[]]

    def process(self, frame):
        if isinstance(frame, _core.Frame):
            self._storage[-1].append(frame)
            if isinstance(frame, _core.GateCloseFrame):
                self._storage.append([])
        return frame

    @property
    def recorded_data(self):
        series = [np.concatenate([frame.frame for frame in series], axis=-1) for series in self._storage if len(series) > 0]
        if len(series) == 1:
            return series[0]
        try:
            return np.stack(series, axis=0)
        except ValueError:
            return series


class ZArrRecorder(_Recorder):
    timeout = 1
    def __init__(
        self,
        filename=None,
        mode='w-',
        chunksize=50000,
        threaded=False,
        **kwargs
    ):
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

    def to_dict(self):
        return super().to_dict() | dict(
            filename=self.filename,
            chunksize=self.chunksize,
            thereaded=self.threaded,
            mode=self.mode,
        )

    def setup(self, **kwargs):
        super().setup(**kwargs)
        if self.threaded:
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self.threaded_writer)
            self._stop_event = threading.Event()
            self._thread.start()
        else:
            self.writer = self.initialize_writer()
            self.writer.send(None)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._is_ready = False

    def process(self, frame):
        if not self._is_ready:
            self.setup()
        if isinstance(frame, _core.Frame):
            if self.threaded:
                self._q.put(frame)
            else:
                try:
                    self.writer.send(frame)
                except StopIteration:
                    pass
        return frame

    def initialize_writer(self):
        self.store = zarr.storage.DirectoryStore(self.filename)

        frame = (yield)
        channels, framesize = frame.frame.shape
        chunksize = self.chunksize

        buffer = np.zeros((channels, chunksize))
        self.data = zarr.open_array(
            store=self.store,
            mode=self.mode,
            shape=(0, channels, 0),
            chunks=(1, 1, chunksize),
        )
        self.data.attrs['samplerate'] = self.samplerate
        self.data.attrs['pipeline'] = self._pipeline.to_dict()
        buffer_idx = 0
        shot_idx = -1
        while not isinstance(frame, _core.LastFrame):
            if frame is None:
                frame = (yield)
            channels, framesize = frame.frame.shape

            if isinstance(frame, _core.GateOpenFrame) or shot_idx == -1:
                assert buffer_idx == 0, "There seems to be data in the buffer at the start of a shot recording!"
                shot_idx += 1
                _shots, _channels, _samples = self.data.shape
                self.data.resize((_shots + 1, _channels, _samples))
                write_idx = 0

            if framesize < chunksize - buffer_idx:
                # This frame can fit in the buffer
                buffer[:, buffer_idx:buffer_idx + framesize] = frame.frame
                buffer_idx += framesize
                if isinstance(frame, _core.GateCloseFrame):
                    # Gate just closed, we should write the frame and then get a new one
                    frame = None
                else:
                    # In the middle of the open segment, just get the next frame without writing to file
                    frame = None
                    continue
            else:
                # The frame will fill the buffer
                buffer[:, buffer_idx:] = frame.frame[:, :chunksize - buffer_idx]
                if chunksize - buffer_idx < framesize:
                    # The remainder is kept to next loop iteration
                    frame.frame = frame.frame[:, chunksize - buffer_idx:]
                else:
                    # The frame perfectly fit in the buffer, so we need a new one the next iteration
                    frame = None
                buffer_idx = chunksize

            # The buffer is full, time to write to file
            if write_idx + buffer_idx >= self.data.shape[-1]:
                # Array is too small to hold the entire shot
                _shots, _channels, _ = self.data.shape
                _samples = write_idx + buffer_idx
                _samples = _samples + (-_samples % chunksize)
                self.data.resize((_shots, _channels, _samples))
            self.data[shot_idx, :, write_idx:write_idx + buffer_idx] = buffer[:, :buffer_idx]
            write_idx += buffer_idx
            buffer_idx = 0

    def zip_store(self):
        zipname = self.filename + '.zip'
        if os.path.exists(zipname):
            if self.mode == 'w':
                os.remove(zipname)
            else:
                raise FileExistsError(f"Cannot zip data to existing file '{zipname}'. Merging recordings is not supported.")
        zip_store = zarr.storage.ZipStore(
            zipname,
            compression=zarr.storage.zipfile.ZIP_DEFLATED,
        )
        zarr.copy_store(self.store, zip_store)
        zarr.storage.rmdir(self.store)
        self.store = zip_store
        self.data = zarr.open_array(self.store)

    def threaded_writer(self):
        writer = self.initialize_writer()
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
