

#%%
import simpy


#%%
""" Monitoring processes """

data = [] # This list will hold all collected data

def test_process(env, data):
    val = 0
    for i in range(5):
        val += env.now
        data.append(val) # Collect data
        yield env.timeout(1)

env = simpy.Environment()
p = env.process(test_process(env, data))
env.run(p)
print('Collected', data) # Lets see what we got

# %%
""" Monitoring resource usage 
how you can add callbacks to a resource that get called just before or after a get / request or a put / release event
#! Monkey patches
"""

from functools import partial, wraps
import simpy

def patch_resource(resource, pre=None, post=None):
    """Patch *resource* so that it calls the callable *pre* before each
    put/get/request/release operation and the callable *post* after each
    operation. The only argument to these functions is the resource
    instance.
    ...
    """
    def get_wrapper(func):
        # Generate a wrapper for put/get/request/release
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is the actual wrapper
            # Call "pre" callback
            if pre:
                pre(resource)

            # Perform actual operation
            ret = func(*args, **kwargs)

            # Call "post" callback
            if post:
                post(resource)

            return ret
        return wrapper

    # Replace the original operations with our wrapper
    for name in ['put', 'get', 'request', 'release']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))

def monitor(data, resource):
    """This is our monitoring callback."""
    item = (
        resource._env.now, # The current simulation time
        resource.count, # The number of users
        len(resource.queue), # The number of queued processes
    )
    data.append(item)

def test_process(env, res):
    with res.request() as req:
        yield req
        yield env.timeout(1)

env = simpy.Environment()

res = simpy.Resource(env, capacity=1)
data = []
# Bind *data* as first argument to monitor()
# see https://docs.python.org/3/library/functools.html#functools.partial
monitor = partial(monitor, data)
patch_resource(res, post=monitor) # Patches (only) this resource instance

p = env.process(test_process(env, res))
env.run(p)

print(data)


# %%
""" only want to know how many processes are waiting for a Resource at a time """

class MonitoredResource(simpy.Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def request(self, *args, **kwargs):
        self.data.append((self._env.now, len(self.queue)))
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        self.data.append((self._env.now, len(self.queue)))
        return super().release(*args, **kwargs)

def test_process(env, res):
    with res.request() as req:
        yield req
        yield env.timeout(1)

env = simpy.Environment()

res = MonitoredResource(env, capacity=1)
p1 = env.process(test_process(env, res))
p2 = env.process(test_process(env, res))
env.run()

print(res.data)


# %%
""" Event Tracing """

from functools import partial, wraps
import simpy

def trace(env, callback):
    """Replace the ``step()`` method of *env* with a tracing function
    that calls *callbacks* with an events time, priority, ID and its
    instance just before it is processed.

    """
    def get_wrapper(env_step, callback):
        """Generate the wrapper for env.step()."""
        @wraps(env_step)
        def tracing_step():
            """Call *callback* for the next event if one exist before
            calling ``env.step()``."""
            if len(env._queue):
                t, prio, eid, event = env._queue[0]
                callback(t, prio, eid, event)
            return env_step()
        return tracing_step

    env.step = get_wrapper(env.step, callback)

def monitor(data, t, prio, eid, event):
    data.append((t, eid, type(event)))

def test_process(env):
    yield env.timeout(1)

data = []
# Bind *data* as first argument to monitor()
# see https://docs.python.org/3/library/functools.html#functools.partial
monitor = partial(monitor, data)

env = simpy.Environment()
trace(env, monitor)

p = env.process(test_process(env))
env.run(until=p)

for d in data:
    print(d)



# %%
