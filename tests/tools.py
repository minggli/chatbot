"""
tools

testing tools
"""
import cProfile
import pstats
from functools import wraps


class Profiler(object):
    def __init__(self, io=None):
        self._io = io

    def __call__(self, func, *args, **kwargs):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            rv = pr.runcall(func, *args, **kwargs)
            pr.disable()
            s = pstats.Stats(pr, stream=self._io)
            s = s.strip_dirs().sort_stats('tottime')
            s.print_stats()
            return rv
        return wrapper
