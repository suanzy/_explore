"""This module simulates a single-echelon supply chain
and calculates inventory profile (along with associated inventory 
parameters such as on-hand, inventory position, service level, etc.) 
across time

The system follows a reorder point-reorder quantity policy
If inventory position <= ROP, an order of a fixed reorder 
quantity (ROQ) is placed by the facility

It is assumed that any unfulfilled order is backordered
and is fulfilled whenever the material is available in the
inventory. The service level is estimated based on how
late the order was fulfilled

Demand is assumed to be Normally distributed
Lead time is assumed to follow a uniform distribution
"""



import simpy
import numpy as np

# Stocking facility class
class inventoryFacility(object):

    # initialize the new facility object
    def __init__(self, env, mode, initialInv, ROP, ROQ, meanDemand, demandStdDev, minLeadTime, maxLeadTime):
        self.env = env
        self.on_hand_inventory = initialInv
        self.inventory_position = initialInv
        self.mode = mode
        self.ROP = ROP
        self.ROQ = ROQ
        self.meanDemand = meanDemand
        self.demandStdDev = demandStdDev
        self.minLeadTime = minLeadTime
        self.maxLeadTime = maxLeadTime
        self.totalDemand = 0.0
        self.totalBackOrder = 0.0
        self.totalLateSales = 0.0
        self.totalShipped = 0.0
        self.serviceLevel = 0.0
        env.process(self.runOperation(mode="lost"))

    # main subroutine for facility operation
    # it records all stocking metrics for the facility
    def runOperation(self, mode):
        while True:
            yield self.env.timeout(1.0)
            demand = float(np.random.normal(self.meanDemand, self.demandStdDev, 1))
            self.totalDemand += demand
            if mode == "lost":
                shipment = min(demand, self.on_hand_inventory)
                self.totalShipped += shipment
            if mode == "backorder":
                shipment = min(demand + self.totalBackOrder, self.on_hand_inventory)
                backorder = demand - shipment
                self.totalBackOrder += backorder
                self.totalLateSales += max(0.0, backorder)
            self.on_hand_inventory -= shipment
            self.inventory_position -= shipment
           
            if self.inventory_position <= 1.01 * self.ROP:  # multiply by 1.01 to avoid rounding issues
                self.env.process(self.ship(self.ROQ))
                self.inventory_position += self.ROQ

    # subroutine for a new order placed by the facility
    def ship(self, orderQty):
        leadTime = int(np.random.uniform(self.minLeadTime, self.maxLeadTime, 1))
        yield self.env.timeout(leadTime)  # wait for the lead time before delivering
        self.on_hand_inventory += orderQty


# Simulation module
def simulateNetwork(seedinit, mode, CS, ROQ, meanDemand, demandStdDev, minLeadTime, maxLeadTime):
    env = simpy.Environment()  # initialize SimPy simulation instance
    np.random.seed(seedinit)
    ROP = max(CS,ROQ)
    initialInv = ROP + ROQ
    s = inventoryFacility(env, mode, initialInv, ROP, ROQ, meanDemand, demandStdDev, minLeadTime, maxLeadTime)
    env.run(until=365)  # simulate for 1 year
    if mode == "lost":
        s.serviceLevel = s.totalShipped / s.totalDemand
    if mode == "backorder":
        s.serviceLevel = 1 - s.totalLateSales / s.totalDemand
    return s 


######## Main statements to call simulation ########

# Simulate
n = 100
sL = []
for i in range(n):
    nodes = simulateNetwork(i, mode = "lost", CS = 5000, ROQ = 6000, meanDemand = 500, demandStdDev = 100, minLeadTime =7, maxLeadTime = 13)
    sL.append(nodes.serviceLevel)

sLevel = np.array(sL)
print("Avg. service level: " + str(np.mean(sLevel)))
print("Service level standard deviation: " + str(np.std(sLevel)))

# combine these 2 first
# separate utility