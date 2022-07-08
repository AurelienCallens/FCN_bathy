#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A generator that wraps a Keras Sequence and simulates a `fit_generator`
behavior for custom training loops.

Usage:
    from src.dataloader.seq_iterator_gan import ParallelIterator

Author:
    https://stackoverflow.com/questions/58785715/training-gan-in-keras-with-fit-generator?noredirect=1&lq=1
"""

import numpy as np
import multiprocessing.dummy as mp


def ParallelIterator(keras_sequence, epochs, shuffle, use_on_epoch_end,
                     workers=4, queue_size=10):

    sourceQueue = mp.Queue()                     #queue for getting batch indices
    batchQueue = mp.Queue(maxsize=queue_size)  #queue for getting actual batches 
    indices = np.arange(len(keras_sequence))     #array of indices to be shuffled

    use_on_epoch_end = 'on_epoch_end' in dir(keras_sequence) if use_on_epoch_end == True else False
    batchesLeft = 0

    # Fills the batch indices queue (called when sourceQueue is empty -> a few 
    # batches before an epoch ends)
    def fillSource():
        nonlocal batchesLeft

        if shuffle:
            np.random.shuffle(indices)

        # Puts the indices in the indices queue
        batchesLeft += len(indices)
        for i in indices:
            sourceQueue.put(i)

    # Function that will load batches from the Keras Sequence
    def worker():
        nonlocal sourceQueue, batchQueue, keras_sequence, batchesLeft

        while True:
            index = sourceQueue.get(block = True) # get index from the queue

            if index is None:
                break

            item = keras_sequence[index] # get batch from the sequence
            batchesLeft -= 1

            batchQueue.put((index,item), block=True) #puts batch in the batch queue

    # Creates the thread pool that will work automatically as we get from the
    # batch queue
    pool = mp.Pool(workers, worker)
    fillSource()   # At this point, data starts being taken and stored in the 
    # batchQueue

    # Generation loop
    for epoch in range(epochs):

        # If not waiting for epoch end synchronization, always keeps 1 epoch filled ahead
        if (use_on_epoch_end == False):
            if epoch + 1 < epochs: # Only fill if not last epoch
                fillSource()

        for batch in range(len(keras_sequence)):

            # If waiting for epoch end synchronization, wait for workers to have no batches left to get, then call epoch end and fill
            if use_on_epoch_end == True:
                if batchesLeft == 0:
                    keras_sequence.on_epoch_end()
                    if epoch + 1 < epochs:  # Only fill if not last epoch
                        fillSource()
                    else:
                        batchesLeft = -1   # In the last epoch, prevents from calling epoch end again and again

            # Yields batches for the outside loop that is using this generator
            originalIndex, batchItems = batchQueue.get(block = True)
            yield epoch, batch, originalIndex, batchItems


#         print("iterator epoch end")
#     printQueue.put("closing threads")

    # Terminating the pool - add None to the queue so any blocked worker gets released
    for i in range(workers):
        sourceQueue.put(None)
    pool.terminate()
    pool.close()
    pool.join()


    del pool,sourceQueue,batchQueue
#     del printPool, printQueue