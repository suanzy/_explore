""" 
Pre-requisite: 
1. profit maximizing objective (instead of cost minimizing)
2. flexible demand and price ranges (per product) and fab capacities.
3. vary ASP (average selling point) of any product by customer
4. flexible capacity acrosss the product mix

Optimization case:
1. Fab-specific optimal solution -- specific  fab (in  this  case  study,  Fab2)  would  achieve  minimum  wafer  cost  and  chooses  each  subsequent  variable  following  a  greedy algorithm for the specific stage of the supply chain be fully utilized (i.e., 100% utilization)
2. Globally optimal solution -- global  optimization  without  any additional constraints such as in case (a)

"""

from asyncio import constants
import os
import string
import pandas as pd
import pickle
import json
import random
import numpy as np
from datetime import datetime, timezone

from aaog.logger.decorator import log_this
from aaog.logger.logging_handler import LoggingHandler

_logging = LoggingHandler.get_logger(__name__)

@log_this(_logging)
def demand_matrix(cls,customer_config, product_config): 
    """
    Define the min max demand matrix by each cutomer, follow by each product
    #!no error occur but dunno why it is not replacing the Dmin with the config data
    """
    customer_list = OPT.config_key_list(config=customer_config)
    product_list = OPT.config_key_list(config=product_config)
    Dmin = np.full((len(product_list), len(customer_list)), 0)
    Dmax = np.full((len(product_list), len(customer_list)), 0)
    for i in customer_list:
        for j in product_list:
            for a in range(len(customer_list)):
                for b in range(len(customer_list)):
                    Dmin[a,b] = customer_config[i][j]["demand"]["min"] #* most probably overwritten by the 0 later
                    Dmax[a,b] = customer_config[i][j]["demand"]["max"]
    cls._logging.info(f'Matrix of minimum demand: {Dmin}')
    cls._logging.info(f'Matrix of maximum demand: {Dmax}')
    return Dmin, Dmax

if __name__ == '__main__':
    # LoggingHandler.start_log('testlog') 
    # _log = LoggingHandler.get_logger(__name__)

    # path=r"C:\Users\34389\Documents\Repo\simpy_explore\Skeleton"

    config = UTIL.load_config(path ="C:/Users/34389/Documents/Repo/simpy_explore/Skeleton/config/config.json")

    customer_config = UTIL.load_config(config["customer_config_path"])
    product_config = UTIL.load_config(config["product_config_path"])
    facility_config = UTIL.load_config(config["facility_config_path"])
    initial_config = UTIL.load_config(config["initial_config_path"])

    # OPT.init_decision_var(config, initial_config, customer_config, product_config, facility_config)
    # a = OPT.config_key_list(config=customer_config)
    # print(a)
    OPT.demand_matrix(customer_config=customer_config, product_config=product_config)
    # OPT.check_constraint(initial_config=initial_config, customer_config=customer_config, product_config=product_config, facility_config=facility_config)

 


    # total_x = num_product * num_fab * num_AT
    # total_y = num_product * num_AT * num_customer


    """ AT  production  costs 
    
    Print


    Outcome:
    1. x_ijv : The amount of wafers of product i shipped from fab j to AT v.
    2. y_ivk : The amount of units of product i supplied from AT v to customer k. 
    

    """
    # wca = 123# prod cost per wafer
    # # at_prod_cost = np.dot(np.sum(wca + np.dot(dca,dpw)),np.sum(x))
    # print(a)

    # a =np.array([2,3,4,5,6])
    # b = np.array([1,2,3,4,5])
    # c= a*b
    # print(c)


    # print(facility_config["AT_1"]['cost_kdie']) # return the value
    # print(facility_config["AT_1"]['cost_shipment'])
    # print(facility_config["AT_1"]['cost_shipment'].items()) # return dictionary item
    # a = facility_config["AT_1"]['cost_shipment'].items()
    # for key, value in a:
    #     print(key)
