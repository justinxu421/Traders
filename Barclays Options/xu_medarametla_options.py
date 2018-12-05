'''
currently displays "Websocket connection has closed" after trying to buy/sell any option
unsure why, but the example is crashing as well. Will try to fix before competition
'''

import tradersbot as tt
import pprint
import random
import sys
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

t = tt.TradersBot(host=sys.argv[1], id=sys.argv[2], password=sys.argv[3])
pp = pprint.PrettyPrinter(indent=4)

all_prices = defaultdict(list)
realized_vols = {}
realized_prices = {}
delta_price = {}
sharpe_ratios = {}
deltas = {}
total_delta = 0
arr_of_props = []

# Keeps track of prices
SECURITIES = {}
SECURITIES_BUY = {}
SECURITIES_SELL = {}
ALL_ORDERS = {}
security_names = set()
ticks = 0
final_length = 0

# Initializes the prices
def ack_register_method(msg, order):
	global SECURITIES, final_length, security_names
	security_dict = msg['case_meta']['securities']
	for security in security_dict.keys():
		if not(security_dict[security]['tradeable']): 
			continue
		SECURITIES[security] = security_dict[security]['starting_price']
		SECURITIES_BUY[security] = security_dict[security]['starting_price']
		SECURITIES_SELL[security] = security_dict[security]['starting_price']
		all_prices[security].append(security_dict[security]['starting_price'])
		if security != 'TMXFUT':
			security_names.add(security)

	# security_names = random.sample(security_names, 5)
	#print(SECURITIES)
	final_length = msg['case_meta']['case_length']
	print(security_names)

def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    n = len(values)
    variance = np.average((values-average)**2, weights=weights)*(n/(n-1))
    return math.sqrt(variance)

def get_volatility(data):
	n = len(data)
	if n < 2:
		return -1
	else:
		data = np.array(data)
		deltas = np.log(data[1:]/data[:-1])
		c = 1 + 3/len(deltas)
		weights = [c**i for i in range(len(deltas))]
		#return np.std(deltas, ddof=1)
		return weighted_std(deltas,weights)

		#deltas = data[1:] - data[:-1]
		#squared_deltas = deltas**2
		#return np.sqrt(squared_deltas.mean())
# Updates latest price
# make the trades here
def market_update_method(msg, order):
	# print("MARKET", msg['elapsed_time'])
	# print(msg['market_state']['time'])
	global SECURITIES, ticks
	asset = msg['market_state']['ticker']
	try:
		buy_price = min(list(map(float, msg['market_state']['asks'].keys())))
		sell_price = max(list(map(float, msg['market_state']['bids'].keys())))
	except:
		return	
	SECURITIES_BUY[asset] = buy_price
	SECURITIES_SELL[asset] = sell_price
	SECURITIES[asset] = (buy_price + sell_price)/2

	all_prices[asset].append(msg['market_state']['last_price'])
	data = all_prices[asset]
	realized_vols[asset] = get_volatility(data)
	if msg['market_state']['ticker'] == 'TMXFUT':
		print( msg['market_state']['last_price'])
	# print(pp.pprint(msg))

# Buys or sells in a random quantity every time it gets an update
# You do not need to buy/sell here
# change holding of underlying future (hedging)
def d1(spot, strike, vol, time_till_end):
	d1 = np.log(spot/strike) + (vol**2)/2*time_till_end
	d1 = d1/(vol * np.sqrt(time_till_end))
	return d1

def d2(d1, vol, time_till_end):
	return d1 - vol*np.sqrt(time_till_end)

def trader_update_method_arb(msg, order):
	global SECURITIES, realized_vols, ticks, final_length,delta_price,sharpe_ratios, total_delta, arr_of_props
	ticks += 1

	print("PNL is ", msg['trader_state']['pnl'])
	print("Actual PNL is ", msg['trader_state']['pnl']['USD'] + 1000*ticks)

	arb_amt = 100
	for i in range(80,120):
		if SECURITIES_BUY['T'+str(i)+'C'] < SECURITIES_SELL['T'+str(i+1)+'C']:
			print('buy')
			order.addBuy('T'+str(i)+'C', quantity=arb_amt, price=SECURITIES_BUY['T'+str(i)+'C'])
			order.addSell('T'+str(i+1)+'C', quantity=arb_amt, price = SECURITIES_SELL['T'+str(i+1)+'C'])
		if SECURITIES_BUY['T'+str(i+1)+'P'] < SECURITIES_SELL['T'+str(i)+'P']:
			print('sell')
			order.addBuy('T'+str(i+1)+'P', quantity=arb_amt, price=SECURITIES_BUY['T'+str(i+1)+'P'])
			order.addSell('T'+str(i)+'P', quantity=arb_amt, price=SECURITIES_SELL['T'+str(i)+'P'])

def trader_update_method(msg, order):
	global SECURITIES, realized_vols, ticks, final_length,delta_price,sharpe_ratios, total_delta, arr_of_props
	ticks += 1

	print("PNL is ", msg['trader_state']['pnl'])
	print("Actual PNL is ", msg['trader_state']['pnl']['USD'] + 1000*ticks)

	spot_price = SECURITIES['TMXFUT']
	volatility = realized_vols['TMXFUT']
	cur_time = ticks

	positions = msg['trader_state']['positions']
	

	arb_amt = 100
	for i in range(80,120):
		if SECURITIES_BUY['T'+str(i)+'C'] < SECURITIES_SELL['T'+str(i+1)+'C']:
			order.addBuy('T'+str(i)+'C', quantity=arb_amt, price=SECURITIES_BUY['T'+str(i)+'C'])
			order.addSell('T'+str(i+1)+'C', quantity=arb_amt, price = SECURITIES_SELL['T'+str(i+1)+'C'])
		if SECURITIES_BUY['T'+str(i+1)+'P'] < SECURITIES_SELL['T'+str(i)+'P']:
			order.addBuy('T'+str(i+1)+'P', quantity=arb_amt, price=SECURITIES_BUY['T'+str(i+1)+'P'])
			order.addSell('T'+str(i)+'P', quantity=arb_amt, price=SECURITIES_SELL['T'+str(i)+'P'])


	if ticks > 10:
		total_delta = 0
		count = 0

		c_80_price = SECURITIES['T80C']
		c_120_price = SECURITIES['T120C']
		p_80_price = SECURITIES['T80P']
		p_120_price = SECURITIES['T120P']

		true_price = round(spot_price)
		#print('Sell: ',SECURITIES_SELL['TMXFUT'])
		#print('Buy: ',SECURITIES_BUY['TMXFUT'])
		if true_price < 80:
			true_price = 80
		if true_price > 120:
			true_price = 120
		

		c_true_price = SECURITIES['T'+str(true_price)+'C']
		p_true_price = SECURITIES['T'+str(true_price)+'P']

		delta_edges = {}
		for security in ['T80P','T80C','T120C','T120P','T'+str(true_price)+'C','T'+str(true_price)+'P']:
			is_call =  security.endswith('C')
			amt = int(security[1:-1])
			d_1 = d1(spot_price,amt,volatility,final_length - cur_time)
			d_2 = d2(d_1, volatility, final_length - cur_time)
			value = norm.cdf(d_1)*spot_price - norm.cdf(d_2)*amt
			#print(security,': ',value)
			if not is_call:
				value = value + amt - spot_price
			delta_edges[security] = SECURITIES[security] - value
		
		c_correction = {}
		p_correction = {}
		for i in range(80,121):
			if i < true_price and true_price > 80:
				c_correction[i] = ((i-80)*delta_edges['T'+str(true_price)+'C'] + (true_price-i)*delta_edges['T80C'])/(true_price-80)
				p_correction[i] = ((i-80)*delta_edges['T'+str(true_price)+'P'] + (true_price-i)*delta_edges['T80P'])/(true_price-80)
			if i > true_price and true_price < 120:
				c_correction[i] = ((120-i)*delta_edges['T'+str(true_price)+'C'] + (i-true_price)*delta_edges['T120C'])/(120-true_price)
				p_correction[i] = ((120-i)*delta_edges['T'+str(true_price)+'P'] + (i-true_price)*delta_edges['T120P'])/(120-true_price)
			if i == true_price:
				c_correction[i] = delta_edges['T'+str(true_price)+'C']
				p_correction[i] = delta_edges['T'+str(true_price)+'P']
		count = 0
		
		for security in security_names:
			is_call =  security.endswith('C')
			amt = int(security[1:-1])
			d_1 = d1(spot_price,amt,volatility,final_length - cur_time)
			d_2 = d2(d_1, volatility, final_length - cur_time)
			value = norm.cdf(d_1)*spot_price - norm.cdf(d_2)*amt
			#print(security,': ',value)
			if not is_call:
				value = value + amt - spot_price

			multiplier = 0.8
			if is_call:
				value += multiplier*c_correction[amt]
			else:
				value += multiplier*p_correction[amt]
			
			realized_prices[security] = value

			deltas[security] = norm.cdf(d_1)
			if not is_call:
				deltas[security] -= 1

			delta_price[security] = value - SECURITIES[security]
			#print(security,': ', delta_price[security])
			
			max_amt_hold = 60
			#print(sharpe_ratios[security])
			
			if value>SECURITIES_BUY[security] and positions[security] < max_amt_hold:
				print('buy')
				print(max_amt_hold + positions[security])
				order.addBuy(security, quantity=max_amt_hold - positions[security], price=SECURITIES_BUY[security])
				total_delta += (max_amt_hold - positions[security])*deltas[security]
			if value<SECURITIES_SELL[security] and positions[security] > (-1)*max_amt_hold:
				print('sell')
				print(max_amt_hold + positions[security])
				order.addSell(security, quantity=max_amt_hold + positions[security],price=SECURITIES_SELL[security])
				total_delta -= (max_amt_hold + positions[security])*deltas[security]

		if ticks%5 == 0:
			curr_future_amt = positions['TMXFUT']
			order.addSell('TMXFUT', int(curr_future_amt + total_delta))

def ack_modify_orders_method(msg, order):
	global ticks
	for order_msg in msg['orders']:
		ALL_ORDERS[order_msg['order_id']] = (order_msg['ticker'], ticks)

###############################################
#### You can add more of these if you want ####
###############################################

t.onAckRegister = ack_register_method
t.onMarketUpdate = market_update_method
t.onTraderUpdate = trader_update_method
#t.onTrade = trade_method
t.onAckModifyOrders = ack_modify_orders_method
#t.onNews = news_method
t.run()