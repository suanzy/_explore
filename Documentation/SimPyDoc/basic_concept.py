"""
Environment -- Create instance
Environment.timeout() -- trigger after a certain amount of (simulated) time has passed.
Environment.process() -- generate process, process interaction
Environment.run() -- passing an end time
"""

""" Example:
Car Process - car will alternately drive and park for a while """

def car(env):
    while True:
        print('Start parking at %d' % env.now)
        parking_duration = 5
        yield env.timeout(parking_duration)
        
        print('Start driving at %d' % env.now)
        trip_duration = 2
        yield env.timeout(trip_duration)

# import random

# random.seed(500)
# print(random.random())

import simpy
env = simpy.Environment()
env.process(car(env))
env.run(until=15)

