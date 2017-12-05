
import numpy as np
from time import time
from npstreams import isum, last

try:
    from npstreams.cuda import csum
except ImportError:
    WITH_CUDA = False
else:
    WITH_CUDA = True

if __name__ == '__main__':

    print('Benchmarking...')
    stream = [np.random.random((2048, 2048)) for _ in range(50)]

    start = time()
    s = np.sum(np.stack(stream, axis = -1), axis = 2)
    delay = time() - start
    print('numpy.sum and numpy.stack: ', delay, 's')

    stack = np.stack(stream, axis = -1)
    start = time()
    s = np.sum(stack, axis = 2)
    delay = time() - start
    print('numpy.sum on existing stack: ', delay, 's')

    start = time()
    s = sum(stream)
    delay = time() - start
    print('Builtin sum: ', delay, 's')

    start = time()
    s = last(isum(stream))
    delay = time() - start
    print('npstreams.isum: ', delay, 's')

    if WITH_CUDA:
        start = time()
        s = csum(stream)
        delay = time() - start
        print('npstreams.cuda.csum: ', delay, 's')