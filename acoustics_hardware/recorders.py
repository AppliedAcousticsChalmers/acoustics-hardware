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
    def __init__(self, filename=None, mode='w-', chunksize=50000, threaded=False, series_written_message=False, chunk_written_message=False, **kwargs):
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
        self.series_written_message = series_written_message
        self.chunk_written_message = chunk_written_message
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
        if isinstance(frame, _core.Frame):
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
        storage = zarr.hierarchy.open_group(self.filename, mode=self.mode)

        frame = (yield)
        channels, framesize = frame.frame.shape
        n_series = 0
        z = None
        while not isinstance(frame, _core.LastFrame):
            channels, framesize = frame.frame.shape
            if z is None:
                z = storage.create(
                    name=timestamps.timestamp('filename', microseconds=True),
                    shape=(channels, 0), chunks=(1, chunksize), dtype=frame.frame.dtype,
                    compressor=None,
                )
                buffer = np.zeros(shape=(channels, chunksize), dtype=frame.frame.dtype)
                write_idx = 0
                z.attrs['samplerate'] = self.samplerate

            if isinstance(frame, _core.GateCloseFrame):
                # Gate just closed on this frame,
                # write what's in the buffer as well this frame and close the array store
                z.append(buffer[:, :write_idx], axis=1)
                z.append(frame.frame, axis=1)
                n_series += 1
                if self.series_written_message:
                    print(f'Finished time series no. {n_series} with {z.shape[1] / self.samplerate:.3f} seconds', flush=True)
                z = None  # Indicates that we should start a new series
                frame = (yield)
            elif framesize < chunksize - write_idx:
                # We can write the entire frame to the buffer
                buffer[:, write_idx:write_idx + framesize] = frame.frame
                write_idx += framesize
                # Get a new frame
                frame = (yield)
            else:
                # The current frame will fill the buffer
                buffer[:, write_idx:] = frame.frame[:, :chunksize - write_idx]
                # Write and reset the buffer, keep the rest of the frame
                z.append(buffer, axis=1)
                if self.chunk_written_message:
                    print(f'{z.shape[1] / self.samplerate:.3f} seconds ({z.shape[1]} samples) recorded in this series', flush=True)
                frame.frame = frame.frame[:, chunksize - write_idx:]
                write_idx = 0

        if z is not None:
            # We got a last frame, which might contain data.
            # If the series is still open, we should write the data remaining in the buffer,
            # as well as the data in the last frame.
            z.append(buffer[:, :write_idx], axis=1)
            z.append(frame.frame, axis=1)
        elif frame.frame.size:
            raise RuntimeError('Got a last frame which has data but the recording series is already closed. This should never happen...')
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
