#%%
import simpy
#%%
def example(env):
    event = simpy.events.Timeout(env, delay=1, value=42)
    value = yield event
    print('now=%d, value=%d' % (env.now, value))

env = simpy.Environment()
example_gen = example(env)
p = simpy.events.Process(env, example_gen)

env.run()


#%%

""" Best Practice -- For the same script above"""

def example(env):
    value = yield env.timeout(1, value=42)
    print('now=%d, value=%d' % (env.now, value))

env = simpy.Environment()
p = env.process(example(env))
env.run()
# %%
