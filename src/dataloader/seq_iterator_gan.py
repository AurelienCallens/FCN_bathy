#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function taken from : https://stackoverflow.com/questions/58785715/training-gan-in-keras-with-fit-generator?noredirect=1&lq=1
"""
import numpy as np
import multiprocessing.dummy as mp

#A generator that wraps a Keras Sequence and simulates a `fit_generator` behavior for custom training loops
#It will also work with any iterator that has `__len__` and `__getitem__`.    
def ParallelIterator(keras_sequence, epochs, shuffle, use_on_epoch_end, workers = 4, queue_size = 10):

    sourceQueue = mp.Queue()                     #queue for getting batch indices
    batchQueue = mp.Queue(maxsize = queue_size)  #queue for getting actual batches 
    indices = np.arange(len(keras_sequence))     #array of indices to be shuffled

    use_on_epoch_end = 'on_epoch_end' in dir(keras_sequence) if use_on_epoch_end == True else False
    batchesLeft = 0

#     printQueue = mp.Queue()                      #queue for printing messages
#     import threading
#     screenLock = threading.Semaphore(value=1)
#     totalWorkers= 0

#     def printer():
#         nonlocal printQueue, printing
#         while printing:
#             while not printQueue.empty():
#                 text = printQueue.get(block=True)
#                 screenLock.acquire()
#                 print(text)
#                 screenLock.release()

    #fills the batch indices queue (called when sourceQueue is empty -> a few batches before an epoch ends)
    def fillSource():
        nonlocal batchesLeft

#         printQueue.put("Iterator: fill source - source qsize = " + str(sourceQueue.qsize()))
        if shuffle == True:
            np.random.shuffle(indices)

        #puts the indices in the indices queue
        batchesLeft += len(indices)
#         printQueue.put("Iterator: batches left:" + str(batchesLeft))
        for i in indices:
            sourceQueue.put(i)

    #function that will load batches from the Keras Sequence
    def worker():
        nonlocal sourceQueue, batchQueue, keras_sequence, batchesLeft
#         nonlocal printQueue, totalWorkers
#         totalWorkers += 1
#         thisWorker = totalWorkers

        while True:
#             printQueue.put('Worker: ' + str(thisWorker) + ' will try to get item')
            index = sourceQueue.get(block = True) #get index from the queue
#             printQueue.put('Worker: ' + str(thisWorker) + ' got item ' +  str(index) + " - source q size = " + str(sourceQueue.qsize()))

            if index is None:
                break

            item = keras_sequence[index] #get batch from the sequence
            batchesLeft -= 1
#             printQueue.put('Worker: ' + str(thisWorker) + ' batches left ' + str(batchesLeft))

            batchQueue.put((index,item), block=True) #puts batch in the batch queue
#             printQueue.put('Worker: ' + str(thisWorker) + ' added item ' + str(index) + ' - queue: ' + str(batchQueue.qsize()))

#         printQueue.put("hitting end of worker" + str(thisWorker))

#       #printing pool that will print messages from the print queue
#     printing = True
#     printPool = mp.Pool(1, printer)

    #creates the thread pool that will work automatically as we get from the batch queue
    pool = mp.Pool(workers, worker)
    fillSource()   #at this point, data starts being taken and stored in the batchQueue

    #generation loop
    for epoch in range(epochs):

        #if not waiting for epoch end synchronization, always keeps 1 epoch filled ahead
        if (use_on_epoch_end == False):
            if epoch + 1 < epochs: #only fill if not last epoch
                fillSource()

        for batch in range(len(keras_sequence)):

            #if waiting for epoch end synchronization, wait for workers to have no batches left to get, then call epoch end and fill
            if use_on_epoch_end == True:
                if batchesLeft == 0:
                    keras_sequence.on_epoch_end()
                    if epoch + 1 < epochs:  #only fill if not last epoch
                        fillSource()
                    else:
                        batchesLeft = -1   #in the last epoch, prevents from calling epoch end again and again

            #yields batches for the outside loop that is using this generator
            originalIndex, batchItems = batchQueue.get(block = True)
            yield epoch, batch, originalIndex, batchItems


#         print("iterator epoch end")
#     printQueue.put("closing threads")

    #terminating the pool - add None to the queue so any blocked worker gets released
    for i in range(workers):
        sourceQueue.put(None)
    pool.terminate()
    pool.close()
    pool.join()
#     printQueue.put("terminated")

#     printing = False
#     printPool.terminate()
#     printPool.close()
#     printPool.join()


    del pool,sourceQueue,batchQueue
#     del printPool, printQueue