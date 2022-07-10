import simpy
import numpy as np

from aaog.logger.decorator import log_this
from aaog.logger.logging_handler import LoggingHandler

# Stocking facility class
class Inventory:

    _logging = LoggingHandler.get_logger(__qualname__)

    # initialize the new facility object
    # def __init__(cls, env, mode, initialInv, ROP, ROQ, meanDemand, demandStdDev, minLeadTime, maxLeadTime):
    #     env = env
    #     on_hand_inventory = initialInv
    #     inventory_position = initialInv
    #     mode = mode
    #     ROP = ROP
    #     ROQ = ROQ
    #     meanDemand = meanDemand
    #     demandStdDev = demandStdDev
    #     minLeadTime = minLeadTime
    #     maxLeadTime = maxLeadTime
    #     totalDemand = 0.0
    #     totalBackOrder = 0.0
    #     totalLateSales = 0.0
    #     totalShipped = 0.0
    #     serviceLevel = 0.0
    #     env.process(runOperation(mode="lost"))

    # 
    # it records all stocking metrics for the facility

    @classmethod
    @log_this(_logging)
    def operation(cls, env, mode, initialInv, ROP, ROQ, meanDemand, demandStdDev, totalDemand, totalBackOrder, totalLateSales, totalShipped, on_hand_inventory, inventory_position, *arg, **kwarg):

        while True:
            # yield env.timeout(1.0)
            demand = float(np.random.normal(meanDemand, demandStdDev, 1))
            cls._logging.info("Demand: %s", str(demand))
            totalDemand += demand
            cls._logging.info("Total Demand: %s", str(totalDemand))
            if mode == "lost":
                shipment = min(demand, on_hand_inventory)
                cls._logging.info("Current Shipment: %s", str(shipment))
                totalShipped += shipment
                cls._logging.info("Total shipment that need to be shipped: %s", str(totalShipped))
            if mode == "backorder":
                shipment = min(demand + totalBackOrder, on_hand_inventory)
                cls._logging.info("Current Shipment: %s", str(shipment))
                backorder = demand - shipment
                cls._logging.info("Current Back Order: %s", str(backorder))
                totalBackOrder += backorder
                cls._logging.info("Total Back Order: %s", str(totalBackOrder))
                totalLateSales += max(0.0, backorder)
                cls._logging.info("Total Late Sales: %s", str(totalLateSales))
            on_hand_inventory -= shipment
            cls._logging.info("On Hand Inventory after: %s", str(on_hand_inventory))
            inventory_position -= shipment
            cls._logging.info("Inventory Health: %s", str(inventory_position))
           
            if inventory_position <= 1.01 * ROP:  # multiply by 1.01 to avoid rounding issues
                cls._logging.info("Inventory Health is less then ROP, initiate new shipment order.")
                env.process(Inventory.ship(ROQ))
                inventory_position += ROQ
                cls._logging.info("Inventory Health: %s", str(inventory_position))

    # subroutine for a new order placed by the facility
    @classmethod
    @log_this(_logging)
    def ship(cls, env, orderQty, minLeadTime, maxLeadTime):
        leadTime = int(np.random.uniform(minLeadTime, maxLeadTime, 1))
        cls._logging.info("Shipping leadtime: %s", str(leadTime))
        yield env.timeout(leadTime)  # wait for the lead time before delivering
        on_hand_inventory += orderQty
        cls._logging.info("On hand inventory: %s", str(on_hand_inventory))


# Simulation module
class Simulate(object):

    _logging = LoggingHandler.get_logger(__qualname__)

    @classmethod
    @log_this(_logging)
    def network(cls, seedinit, mode, CS, ROQ, meanDemand, demandStdDev, *arg, **kwarg):
        env = simpy.Environment()  # initialize SimPy simulation instance
        np.random.seed(seedinit)
        ROP = max(CS,ROQ)
        initialInv = ROP + ROQ
        totalDemand = 0.0
        totalBackOrder = 0.0
        totalLateSales = 0.0
        totalShipped = 0.0
        serviceLevel = 0.0
        on_hand_inventory = initialInv
        inventory_position = initialInv
        cls._logging.info("Initial Inventory: %s", str(initialInv))
        s = env.process(Inventory.operation(env, mode, initialInv, ROP, ROQ, meanDemand, demandStdDev, totalDemand, totalBackOrder, totalLateSales, totalShipped, on_hand_inventory, inventory_position))
        # s = Inventory.operation(env, mode, initialInv, ROP, ROQ, meanDemand, demandStdDev)
        print(s.totalShipped)
        env.run(until=365)  # simulate for 1 year
        if mode == "lost":
            s.serviceLevel = s.totalShipped / s.totalDemand
        if mode == "backorder":
            s.serviceLevel = 1 - s.totalLateSales / s.totalDemand
        return s 


######## Main statements to call simulation ########

if __name__ == "__main__":

    LoggingHandler.start_log('testlog') 
    _log = LoggingHandler.get_logger(__name__) 

    n = 100
    sL = []
    for i in range(n):
        nodes = Simulate.network(seedinit=i, mode = "lost", CS = 5000, ROQ = 6000, meanDemand = 500, demandStdDev = 100, minLeadTime =7, maxLeadTime = 13)
        _log.info("Service level in run %d: %.2f%%", (i,100*nodes.serviceLevel))
        sL.append(nodes.serviceLevel)

    sLevel = np.array(sL)
    print("Avg. service level: " + str(np.mean(sLevel)))
    print("Service level standard deviation: " + str(np.std(sLevel)))


# separate utility