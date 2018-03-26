import numpy as np
import queue


def flush_Q(q):
    while True:
        try:
            q.get(timeout=0.1)
        except queue.Empty:
            break


def concatenate_Q(q):
    data_list = []
    while True:
        try:
            data_list.append(q.get(timeout=0.1))
        except queue.Empty:
            break
    return np.concatenate(data_list, axis=-1)
