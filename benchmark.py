
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

    stream = [np.random.random((2048, 2048)) for _ in range(50)]

    print('numpy.sum and numpy.stack:')
    start = time()

    s = np.sum(np.stack(stream, axis = -1), axis = 2)
    delay = time() - start
    print(delay, 's')

    print('Python sum: ')
    start = time()

    s = sum(stream)
    delay = time() - start
    print(delay, 's')

    print('npstreams.isum: ')
    start = time()

    s = last(isum(stream))
    delay = time() - start
    print(delay, 's')

    if WITH_CUDA:
        print('npstreams.cuda.csum: ')
        start = time()

        s = csum(stream)
        delay = time() - start
        print(delay, 's')