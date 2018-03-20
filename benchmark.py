
from timeit import default_timer as timer

import numpy as np

from npstreams import average, iaverage, isum, last
from npstreams import sum as ns_sum

if __name__ == '__main__':

    print('Benchmarking sums ...')
    stream = [np.random.random((2048, 2048)) for _ in range(50)]
    stack = np.stack(stream, axis = -1)

    start = timer()
    s = np.sum(np.stack(stream, axis = -1), axis = 2)
    delay = timer() - start
    print('numpy.sum and numpy.stack: '.ljust(30), '{:.6f} s'.format(delay))

    
    start = timer()
    s = np.sum(stack, axis = 2)
    delay = timer() - start
    print('numpy.sum on existing stack: '.ljust(30), '{:.6f} s'.format(delay))

    start = timer()
    s = sum(stream)
    delay = timer() - start
    print('Builtin sum: '.ljust(30), '{:.6f} s'.format(delay))

    start = timer()
    s = last(isum(stream))
    delay = timer() - start
    print('npstreams.isum: '.ljust(30), '{:.6f} s'.format(delay))

    start = timer()
    s = ns_sum(stream)
    delay = timer() - start
    print('npstreams.sum: '.ljust(30), '{:.6f} s'.format(delay))

    print('\nBenchmarking averages...')

    start = timer()
    s = last(iaverage(stream))
    delay = timer() - start
    print('Via iaverage: '.ljust(30), '{:.6f} s'.format(delay))

    start = timer()
    s = average(stream)
    delay = timer() - start
    print('Via average: '.ljust(30), '{:.6f} s'.format(delay))

    start = timer()
    s = np.average( np.stack(stream, axis = -1), axis = 2)
    delay = timer() - start
    print('numpy.average + numpy.stack: '.ljust(30), '{:.6f} s'.format(delay), '\n')
