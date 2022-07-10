#%%
import time
import simpy

#%%
def example(env):
    start = time.perf_counter()
    yield env.timeout(1)
    end = time.perf_counter()
    print('Duration of one simulation time unit: %.2fs' % (end - start))

env = simpy.Environment()
proc = env.process(example(env))
env.run(until=proc)

#%%
import simpy.rt
env = simpy.rt.RealtimeEnvironment(factor=0.1)
proc = env.process(example(env))
env.run(until=proc)

# %%
""" If the strict parameter is set to True (the default), the step() and run() methods will raise a RuntimeError
if the computation within a simulation time step take more time than the real-time factor allows. """

import time
import simpy.rt

def slow_proc(env):
    time.sleep(0.02) # Heavy computation :-)
    yield env.timeout(1)

env = simpy.rt.RealtimeEnvironment(factor=0.01, strict=False) 
proc = env.process(slow_proc(env))
try:
    env.run(until=proc)
    print('Everything alright')
except RuntimeError:
    print('Simulation is too slow')


# %%
