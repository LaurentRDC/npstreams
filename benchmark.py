
import numpy as np
from timeit import default_timer as timer
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

    start = timer()
    s = np.sum(np.stack(stream, axis = -1), axis = 2)
    delay = timer() - start
    print('numpy.sum and numpy.stack: ', delay, 's')

    stack = np.stack(stream, axis = -1)
    start = timer()
    s = np.sum(stack, axis = 2)
    delay = timer() - start
    print('numpy.sum on existing stack: ', delay, 's')

    start = timer()
    s = sum(stream)
    delay = timer() - start
    print('Builtin sum: ', delay, 's')

    start = timer()
    s = last(isum(stream))
    delay = timer() - start
    print('npstreams.isum: ', delay, 's')

    if WITH_CUDA:
        start = timer()
        s = csum(stream)
        delay = timer() - start
        print('npstreams.cuda.csum: ', delay, 's')