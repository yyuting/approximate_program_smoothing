
"""
Run a Python function in a separate subprocess with limited time and memory.

The psutil and pathos packages are required to handle the processes. Time and memory limits are not strict guarantees, but rather periodically checked by polling.

Example:

>>> def sum_squares(n):
...   return sum(i**2 for i in range(n))
... 
>>> t = time_memory_limiter.TimeMemoryLimiter()
>>> t.run(sum_squares, (1000,), 0.1)
332833500
>>> t.run(sum_squares, (10**7,), 1.0)
>>> # The run() method returns None if the limits are hit.

See the TimeMemoryLimiter.run() method for additional arguments and documentation.
"""

import threading
import time
import psutil
import os
import multiprocessing, multiprocessing.pool
import pathos.multiprocessing

prevent_daemons = True        # Whether to work around bug caused by processes being daemonic.
                              # If False, then TimeMemoryLimiter cannot be used inside a process pool.
                              # If True, then the workaround depends on internals of multiprocessing/pathos,
                              # so it might require further patches as code internals evolve.

if prevent_daemons:
    class NoDaemonProcess(multiprocessing.Process):
        def _get_daemon(self):
            return False
        def _set_daemon(self, v):
            pass
        
        daemon = property(_get_daemon, _set_daemon)

    Pool = pathos.multiprocessing.Pool
    class NoDaemonPool(Pool):
        Process = NoDaemonProcess
        def __init__(self, *args, **kw):
            Pool.__init__(self, *args, **kw)

    pathos.multiprocessing.Pool = NoDaemonPool

class TimeMemoryLimiter:
    def __init__(self):
        self.pool = pathos.multiprocessing.ProcessingPool(1)

    def run(self, f, args, time_limit=None, memory_limit=None, sleep_time=0.01, stop_message=None, stop_callback=None):
        """
        Run f(*args) with a given time_limit (in seconds) and memory limit (in bytes).
        
        Here sleep_time is the time to sleep between checks for f() finishing or the limits being hit, stop_message is a message string to print if the limits were hit, and stop_callback is called with no arguments if the limits were hit.
        """
        T0 = time.time()
        
        pid = self.pool.pipe(os.getpid)
        process = psutil.Process(os.getpid())
        ans = self.pool.apipe(f, *args)
        while True:
            if ans.ready():
                return ans.get()
            if (time_limit is not None and time.time() - T0 > time_limit) or (memory_limit is not None and process.memory_info().rss > memory_limit):
                if stop_message is not None:
                    print(stop_message)
                if stop_callback is not None:
                    stop_callback()
                self.pool.terminate()
                self.pool.restart()
                return None
            time.sleep(sleep_time)
