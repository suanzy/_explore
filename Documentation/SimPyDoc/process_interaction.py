""" From Previous:
Environment -- Create instance
Environment.timeout() -- trigger after a certain amount of (simulated) time has passed.
Environment.process() -- generate process, process interaction
Environment.run() -- passing an end time

"""
#%%
import simpy
from random import seed, randint
seed(23)

#%%

""" Sleep until woken up """

class Car(object):
    def __init__(self, env):
        self.env = env
        # Start the run process everytime an instance is created.
        self.action = env.process(self.run())

    def run(self):
        while True:
            print('Start parking and charging at %d' % self.env.now)
            charge_duration = 5
            # We yield the process that process() returns to wait for it to finish
            yield self.env.process(self.charge(charge_duration))

            # The charge process has finished and we can start driving again.
            print('Start driving at %d' % self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)

    def charge(self, duration):
        yield self.env.timeout(duration)

env = simpy.Environment()
car = Car(env)
env.run(until=15)


#%%
""" 
Waiting for another process to terminate
"""

def driver(env, car):
    yield env.timeout(3)
    car.action.interrupt()

class Car(object):
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())

    def run(self):
        while True:
            print('Start parking and charging at %d' % self.env.now)
            charge_duration = 5
            # We may get interrupted while charging the battery
            try:
                yield self.env.process(self.charge(charge_duration))
            except simpy.Interrupt:
                # When we received an interrupt, we stop charging and switch to the "driving" state
                print('Was interrupted. Hope, the battery is full enough ')

            print('Start driving at %d' % self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)

    def charge(self, duration):
        yield self.env.timeout(duration)

env = simpy.Environment()
car = Car(env)
env.process(driver(env, car))
env.run(until=15)
# %%
""" Interrupting another process """
class EV:
    def __init__(self, env):
        self.env = env
        self.drive_proc = env.process(self.drive(env))

    def drive(self, env):
        while True:
            # Drive for 20-40 min
            yield env.timeout(randint(20, 40))

            # Park for 1 hour
            print('Start parking at', env.now)
            charging = env.process(self.bat_ctrl(env))
            parking = env.timeout(60)
            yield charging | parking
            if not charging.triggered:
                # Interrupt charging if not already done.
                charging.interrupt('Need to go!')
            print('Stop parking at', env.now)

    def bat_ctrl(self, env):
        print('Bat. ctrl. started at', env.now)
        try:
            yield env.timeout(randint(60, 90))
            print('Bat. ctrl. done at', env.now)
        except simpy.Interrupt as i:
            # Onoes! Got interrupted before the charging was done.
            print('Bat. ctrl. interrupted at', env.now, 'msg:',
                  i.cause)

env = simpy.Environment()
ev = EV(env)
env.run(until=100)


# %%
