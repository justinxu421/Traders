from tradersbot import *
import sys
import pprint
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

#Initialize variables: positions, expectations, future customer orders, etc
position_limit = 5000
case_length = 450
cash = 0
position_lit = 0
position_dark = 0
time = 0
# the current time step
ticks = 0.0
topBid = 0
topAsk = 0

prev_light_price = 0
prev_dark_price = 0
# stores {customer: delta in price)
dark_price_delta = defaultdict(list)
C = 1./25000
#etc etc

market_state = {}
market_state['LIT'] = {'bids': {}, 'asks': {}, 'last_price': 0}
market_state['DARK'] = {'bids': {}, 'asks': {}, 'last_price': 0}
completed_trades = set()

customers = {} # have order history, delta price/quantity after order

curr_customer = ''
curr_amt = 0

open_orders = {}

pp = pprint.PrettyPrinter(indent=4)

def register(msg, TradersOrder):
    #Set case information
    global case_length, ticks

    ticks = 300

    pp.pprint(msg)
    case_length = msg['case_meta']['case_length']
    security_dict = msg['case_meta']['securities']
    prev_light_price = security_dict['TRDRS.LIT']['starting_price']
    prev_dark_price = security_dict['TRDRS.DARK']['starting_price']
    for customer in msg['case_meta']['news_sources'].keys():
        customers[customer] = {'order_history': [], 'delta_price': []}

def update_market(msg, TradersOrder):
    #Update market information
    global ticks
    # pp.pprint(msg)
    print(ticks)
    ticker = msg['market_state']['ticker']
    bids, asks, last_price = msg['market_state']['bids'], msg['market_state']['asks'], msg['market_state']['last_price']
    time = msg['market_state']['time']
    ticks += 0.5
    if ticker == "TRDRS.LIT":
        market_state['LIT']['bids'] = bids
        market_state['LIT']['asks'] = asks
        market_state['LIT']['last_price'] = last_price
    if ticker == "TRDRS.DARK":
        market_state['DARK']['bids'] = bids
        market_state['DARK']['asks'] = asks
        if market_state['DARK']['last_price'] > 0 and market_state['DARK']['last_price'] != last_price:
            dark_price_delta[curr_customer].append((ticks, abs(last_price - market_state['LIT']['last_price'])))
            print(ticks, last_price, market_state['LIT']['last_price'], abs(last_price - market_state['LIT']['last_price']))
        market_state['DARK']['last_price'] = last_price

    # TradersOrder.addBuy("TRDRS.LIT", quantity=1, price=200)

def update_trader(msg, TradersOrder):
    #Update positions
    global position_lit, position_dark
    pnl = msg['trader_state']['pnl']
    positions = msg['trader_state']['positions']
    position_lit = positions['TRDRS.LIT']
    position_dark = positions['TRDRS.DARK']
    open_orders = msg['trader_state']['open_orders']
    cash = msg['trader_state']['cash']
    print('pnl is', pnl, 'cash is', cash, 'positions', positions)

# don't rely on the information from this function
def update_trade(msg, TradersOrder):
    #Update trade information
    pass

def update_order(msg, TradersOrder):
    #Update order information
    global position_lit, position_dark

    for order in msg['orders']:
        ticker = order['ticker']
        buying = order['buy']
        quantity = order['quantity']
        price = order['price']
        sign = 1
        if not buying:
            sign = -1

def update_news(msg, TradersOrder):
    #Update news information
    global prev_light_price, prev_dark_price, position_lit, position_dark, curr_customer, curr_amt, ticks

    headline = msg['news']['headline'].split()
    ticks = msg['news']['time']

    curr_customer = msg['news']['source']
    curr_amt = int(msg['news']['body'])
    print(curr_amt)
    
    price = market_state['LIT']['last_price']
    print(price)

    # if we have excess because trades didn't go through, sell or buy lit
    excess = position_lit + position_dark
    print('excess is', excess)

    dark_amt = 2000
    delta = 0
    if ticks < 15:
        delta = 89.9 
    else:
        sum = 0.
        count = 0.
        for customer in dark_price_delta:
            # get most recent pricing delta
            timestamp, other_delta = dark_price_delta[customer][-1]
            sum += 1./ (ticks - timestamp) * other_delta
            count += 1./ (ticks - timestamp)
            if customer == curr_customer:
                sum += 1./ (ticks - timestamp) * other_delta
                count += 1./ (ticks - timestamp)
        if count >= 1:
            delta = (sum/count) * 0.85
        else:
            delta = 5
    print('change in price is ', delta)

    if ticks + 15 >= case_length:
        if excess > 0:
            while excess > 1000:
                TradersOrder.addSell("TRDRS.LIT", quantity=1000)#, price=price-.5)
                excess -= 1000
            TradersOrder.addSell("TRDRS.LIT", quantity=excess)#, price=price-.5)
        elif excess < 0:
            excess = (-1)*excess
            while excess > 1000:
                TradersOrder.addBuy("TRDRS.LIT", quantity=1000)#, price=price-.5)
                excess -= 1000
            TradersOrder.addBuy("TRDRS.LIT", quantity=excess)#, price=price-.5)
    # if customer is selling
    elif headline[2] == "selling":
        quantity = dark_amt + excess
        while quantity > 1000:
            TradersOrder.addSell("TRDRS.LIT", quantity=1000)#, price=price-.5)
            quantity -= 1000
        TradersOrder.addSell("TRDRS.LIT", quantity=quantity)#, price=price-.5)

        quantity_dark = dark_amt
        while quantity_dark > 1000:
            TradersOrder.addBuy("TRDRS.DARK", quantity=1000, price=price-delta)#, price=price-.5)
            quantity_dark -= 1000
        TradersOrder.addBuy("TRDRS.DARK", quantity=quantity_dark, price=price-delta)
        curr_amt *= -1 
    # if customer is buying
    else:
        quantity = dark_amt - excess
        print(quantity)
        while quantity > 1000:
            TradersOrder.addBuy("TRDRS.LIT", quantity=1000)#, price=price-.5)
            quantity -= 1000
        TradersOrder.addBuy("TRDRS.LIT", quantity=quantity)#, price=price+.5)

        quantity_dark = dark_amt
        while quantity_dark > 1000:
            TradersOrder.addSell("TRDRS.DARK", quantity=1000, price=price+delta)#, price=price-.5)
            quantity_dark -= 1000
        TradersOrder.addSell("TRDRS.DARK", quantity=quantity_dark, price=price+delta)

    delta_price = curr_amt*C
    customers[curr_customer]['delta_price'].append(delta_price)
    customers[curr_customer]['order_history'].append(curr_amt)

    # use the real change in price to decide how accurate this consumer is
    real_delta_price = price - prev_light_price 

    prev_light_price = price

def process():
    #Do stuff to trade
    pass

t = TradersBot(host=sys.argv[1], id=sys.argv[2], password=sys.argv[3])

t.onAckRegister = register
t.onMarketUpdate = update_market
t.onTraderUpdate = update_trader
t.onTrade = update_trade
t.onAckModifyOrders = update_order
t.onNews = update_news

t.run()
