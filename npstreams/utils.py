# -*- coding: utf-8 -*-

from contextlib import contextmanager
from functools import wraps
from warnings import resetwarnings, simplefilter, warn


@contextmanager
def contextwarnings(*args, **kwargs):
    simplefilter(*args, **kwargs)
    yield
    resetwarnings()

def deprecated(message):
    """ 
    Decorator factory that warns of deprecation 
    
    Parameters
    ----------
    message : str
        Message will be dressed up with the name of the function.
    
    Returns
    -------
    decorator : callable
    """

    def decorator(func):
        @wraps(func)
        def newfunc(*args, **kwargs):
            full_message = 'Calls to {} deprecated: {}'.format(func.__name__, message)
            with contextwarnings('always', DeprecationWarning):
                warn(full_message, category = DeprecationWarning, stacklevel = 2)
            return func(*args, **kwargs)
    
    return decorator
