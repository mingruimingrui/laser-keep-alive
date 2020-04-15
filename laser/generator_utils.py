# Copyright (c) 2020 mingruimingrui
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helper functions to work with generators"""

import time
import queue
import threading
from typing import Iterable, Generator


def chunk(g: Iterable, size=10000):
    """Formats output of iterable into chunks of equal sizes

    Arguments:
        g {Iterable} -- An iterable preferrably a generator

    Keyword Arguments:
        size {int} -- The size of each chunk (default: {10000})

    Yields:
        List[object] -- A list of items from `g`
    """
    chunk = []
    for item in g:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


def prefetch(g: Iterable, size=1) -> Generator:
    """Prefetch wrapper around a generator/iterable

    Arguments:
        g {Iterable} -- An iterable preferably a generator

    Keyword Arguments:
        size {int} -- The size of the prefetch queue (default: {1})

    Yields:
        object -- items from `g`
    """
    coord = {'done': False}
    prefetch_queue = queue.Queue(size)

    def fill_prefetch_queue():
        for item in g:
            while not coord['done']:
                try:
                    prefetch_queue.put(item, block=True, timeout=0.1)
                except queue.Full:
                    continue
                else:
                    break

            if coord['done']:
                break

        # Wait for prefetch_queue to be completely consumed
        while prefetch_queue.qsize() > 0:
            time.sleep(0.1)
            continue
        coord['done'] = True

    worker = threading.Thread(target=fill_prefetch_queue)
    worker.setDaemon(True)
    worker.start()

    while not coord['done']:
        try:
            item = prefetch_queue.get(block=True, timeout=0.1)
        except queue.Empty:
            continue
        else:
            yield item

    worker.join()
