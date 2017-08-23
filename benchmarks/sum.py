"""
Benchmark of sum equivalents
"""
import numpy as np
from npstreams.cuda import csum
from npstreams import isum, last
from time import time

def stream():
    for _ in range(50):
        yield np.random.random((2048, 2048)).astype(np.float32)

if __name__ == '__main__':

    print('numpy.sum and numpy.stack:')
    start = time()

    s = np.sum(np.stack(list(stream()), axis = -1), axis = 2)
    delay = time() - start
    print(delay, 's')

    print('npstreams.isum: ')
    start = time()

    s = last(isum(stream()))
    delay = time() - start
    print(delay, 's')

    print('npstreams.cuda.csum: ')

    start = time()

    s = csum(stream())
    delay = time() - start
    print(delay, 's')