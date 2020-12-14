import tensorflow as tf
import numpy as np
import pandas as pd
import random

from utils.data import Data ################
from population.population import Population
from population.network import Network

# suppress tf GPU logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = Data("data/coinbase-1min.csv", interval=30) #################

# for evaluating model fitness
def calculate_profit(trades, prices):
    btc_wallet, starting_cash = 0., 100.
    usd_wallet = starting_cash
    fee = 0.0011

    holding = False
    for idx, trade in enumerate(trades):
        if holding and not np.argmax(trade):
            holding = False
            usd_wallet = btc_wallet * prices[idx] * (1 - fee)
        if not holding and np.argmax(trade):
            holding = True
            btc_wallet = usd_wallet / prices[idx] * (1 - fee)

    # sell if holding
    if holding:
        usd_wallet = btc_wallet * prices[-1] * (1 - fee)

    # discourage models that dont trade
    if usd_wallet == starting_cash:
        return -100.

    return (usd_wallet / starting_cash - 1) * 100

###############
def calc_overperformance(trades, prices):
    btc_wallet, starting_cash = 0., 100.
    usd_wallet = starting_cash
    fee = 0.0011

    num_trades = 0 #
    holding = False
    for idx, trade in enumerate(trades):
        if holding and not np.argmax(trade):
            holding = False
            usd_wallet = btc_wallet * prices[idx] * (1 - fee)
            num_trades += 1
        if not holding and np.argmax(trade):
            holding = True
            btc_wallet = usd_wallet / prices[idx] * (1 - fee)

    # sell if holding
    if holding:
        usd_wallet = btc_wallet * prices[-1] * (1 - fee)

    profit = (usd_wallet / starting_cash - 1) * 100
    holding_ROI = (prices[-1] / prices[0] - 1) * 100

    # discourage models that don't trade
    if num_trades < 5:
        return -100.

    if num_trades > 35:
        return -100.

    if holding_ROI <= 0:
        if profit > 7:
            return 7
        elif profit < 7:
            return profit
    
    if holding_ROI > 0:
        if profit - holding_ROI > 7:
            return 7
        elif profit - holding_ROI < 7:
            return profit - holding_ROI
#################

def get_rand_segment(inputs, prices, size):
    max_idx = len(prices) - size
    rand_idx = np.random.randint(0, max_idx)
    x = inputs[rand_idx:rand_idx + size]
    price = prices[rand_idx:rand_idx + size]

    return x, price


def train_model(layers, 
                inputs, 
                prices,
                pop_size=150,
                data_rotation=0,
                w_mutation_rate = 0.05,
                b_mutation_rate = 0.0,
                mutation_scale = 0.3,
                mutation_decay = 1.,
                reporter=None):
    # network parameters
    network_params = {
        'network': 'feedforward',
        'input': inputs.shape[1],
        'hidden': layers,
        'output': 2
    }

    # build initial population
    pop = Population(network_params,
                     pop_size,
                     mutation_scale,
                     w_mutation_rate,
                     b_mutation_rate,
                     mutation_decay,
                     socket_reporter=reporter)
                     
    g = 1 ###########
    sample_size = 500 # request.json["sampleSize"]
    
    while True:
        if not g % data_rotation: ###########
            ohlc, ta = data.get_rand_segment(sample_size) ##########
            inputs, prices = data.get_training_segment() ##########

        pop.evolve()
        gen_best = pop.run((inputs, prices), fitness_callback=calc_overperformance)
        g += 1
