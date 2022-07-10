
""" A simulation environment manages the simulation time as well as the scheduling and processing of events. """
#%%
import simpy
import progressbar
def my_proc(env):
    yield env.timeout(1)
    return 'Monty Python’s Flying Circus'

env = simpy.Environment()
proc = env.process(my_proc(env))
env.run(until=proc)

#%%
""" If you want to integrate your simulation in a GUI and want to draw a process bar, you can repeatedly call this function with increasing until values and update your progress bar after each cal """
for i in range(100):
    env.run(until=i)
    progressbar.update(i)
# %%
""" peek() returns the time of the next scheduled event or infinity (float('inf')) if no future events are scheduled.
step() processes the next scheduled event. It raises an EmptySchedule exception if no event is available. """

until = 10
while env.peek() < until:
   env.step()

#%%

