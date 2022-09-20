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

class UTIL:

    _logging = LoggingHandler.get_logger(__qualname__)

    @classmethod
    @log_this(_logging)
    def load_config(cls,path):
        f = open(path,)
        # f = open(os.path.join(path, "config","config.json"),)
        cls._logging.info(f'Reading Config File: {path}')
        config = json.load(f)
        # cls._logging.error("Error while reading config file")
        return config

    @classmethod
    @log_this(_logging)
    def save_json(cls,save_path,config):
        # python 2 method
        # f = open(save_path, "w+")
        # f.write(json.dumps(config))
        # f.close()
        # python 3 method
        with open(save_path, 'w') as f:
            json.dump(config, f, ensure_ascii=False)
        cls._logging.info(f'JSON file updated in {save_path}')

class OPT:

    _logging = LoggingHandler.get_logger(__qualname__)


    @classmethod
    @log_this(_logging)
    def range(cls,item, min_value, max_value):
        r = abs(max_value - min_value)
        cls._logging.info(f'Range of {item}: {r}')
        return r
    
    @classmethod
    @log_this(_logging)
    def init_decision_var (cls,config, initial_config, customer_config, product_config, facility_config):
        # Parse list
        customer_list = OPT.config_key_list(config=customer_config)
        product_list = OPT.config_key_list(config=product_config)
        at_list = OPT.config_key_list(config=facility_config['at'])
        fab_list = OPT.config_key_list(config=facility_config['fab'])
        f = initial_config['decision variable']
        # for j in fab_list:
        #     for v in at_list:
        #         for i in product_list:
        #             f['decision variable']['x']['value'][j][v][i] = 0
        #         for k in customer_list:
        #             f['decision variable']['y']['value'][v][k][i] = 0
        # for j in fab_list:
        #     for v in at_list:
        #         for i in product_list:
        #             f['decision variable'['x'['value'[j[v[i]]]]]] = 0
        #         for k in customer_list:
        #             f['decision variable'['y'['value'[v[k[i]]]]]] = 0

        UTIL.save_json(save_path= os.path.join(config["log_path"],"decision_var_0.json") , config = f)

    @classmethod
    @log_this(_logging)
    def constraint(cls, item_value, min_value, max_value):
        if item_value < min_value:
            cls._logging.error(f'{item_value} exceed constraint: less than min value {min_value}')
        elif item_value > max_value:
            cls._logging.error(f'{item_value} exceed constraint: more than max value {max_value}',)
        else:
            cls._logging.info(f'Item checked is within constraint. Value: {item_value}')
  
    @classmethod
    @log_this(_logging)
    def check_constraint(cls,initial_config, customer_config, product_config, facility_config): #* add violation & Penalty later
        # Parse list
        customer_list = OPT.config_key_list(config=customer_config)
        product_list = OPT.config_key_list(config=product_config)
        at_list = OPT.config_key_list(config=facility_config['at'])
        fab_list = OPT.config_key_list(config=facility_config['fab'])
        constraint_list = OPT.config_key_list(config=initial_config["constraint"])

        for t in constraint_list:
            cls._logging.info(f'Checking {t}: {initial_config["constraint"][t]["description"]}')

            # eq 2: Demand range for each product and each customer
            for i in product_list:
                for k in customer_list:
                    # sum_y = 0
                    # for v in at_list: #! v dari mana
                    #     sum_y += initial_config["demand"][k][i] 
                    # a = sum_y
                    a = initial_config["demand"][k][i] #* I think equation 2 is something wrong
                    b = customer_config[k][i]["demand"]["min"]
                    c = customer_config[k][i]["demand"]["max"]
                    OPT.constraint(item_value = a , min_value = b, max_value = c)
            
            # eq 3: Fab capacity range for each fab
            # for j in fab_list:
            #     for i in product_list:
            #         for v in at_list:
            #             a = 
            #             b = facility_config['fab']
            #             c = 
            #             OPT.constraint(item_value = a , min_value = b, max_value = c)
            
            
    # for j in range(num_fab):
    #     sum_CTR_x = 0
    #     for i in range(num_customer):
    #         sum_x = 0
    #         for v in range(num_AT):
    #             sum_x = sum_x + x[i,j,v]
    #         sum_CTR_x = sum_CTR_x + CTR[i] * sum_x



    # @classmethod
    # @log_this(_logging)    
    # def convert_chromosome(chromosome):
    # # Convert the real value to integer
    #     chromosome = [int(item) for item in chromosome]

    #     # Convert the chromosome into value x and y
    #     c_count = 0
    #     for i in range(num_product):
    #         for j in range(num_fab):
    #             for v in range(num_AT):
    #                 x[i,j,v] = chromosome[c_count]
    #                 c_count += 1

    #     for i in range(num_product):
    #         for v in range(num_AT):
    #             for k in range(num_customer):
    #                 y[i,v,k] = chromosome[c_count]
    #                 c_count += 1
    #     return x, y
    
    @classmethod
    @log_this(_logging)
    def config_key_list(cls, config):
        ls=[]
        for key, value in config.items():
            ls.append(key)
        cls._logging.info(f'Config key is listed as: {ls}')        
        return ls

    @classmethod
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

    # @classmethod
    # @log_this(_logging)
    # def check_constraint_1(Dmin= [], num_product=num_product, num_customer=num_customer):
    #     violation = 0
    #     penalty = np.full(5, 0)
    #     for i in range(num_product):
    #         for k in range(num_customer):
    #             sum_y = 0
    #             for v in range(num_AT):
    #                 sum_y += y[i,v,k]
    #             if sum_y < Dmin[i,k]:
    #                 violation += 1
    #                 penalty[0] +=1
    #                 # print("Violation for Equation 2[1] by Product", i, " Customer", k )
    #             elif sum_y > Dmax[i,k]:
    #                 violation += 1
    #                 penalty[0] += 1
    #                 # print("Violation for Equation 2[2] by Product", i, " Customer", k )
    #     return violation, penalty




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
