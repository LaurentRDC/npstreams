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
            full_message = f"""Calls to {func.__name__} deprecated: {message}. 
            {name} will be removed in a future release."""
            with contextwarnings("always", DeprecationWarning):
                warn(full_message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return newfunc

    return decorator
