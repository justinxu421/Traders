from dayof.framework.train_loaders import load_simple, load_medium, load_hard

train_dir = 'train_data'

# Implement your models below
# Test mean square error on the training set by
#   running python test_model.py (possible errors with Python2)
# Depending on your method, you might want to also consider cross 
#   validation or some type of out-of-sample performance metric

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def get_points(a,b):
    return np.array([a,b]).reshape(1,-1)

class SimpleModel(object):
    def __init__(self):
        self.prev_price, self.x1, self.x2, self.next_price = load_simple(train_dir)
        self.train()

    def train(self):
        # train model here
        features = np.stack((self.x1, self.x2), axis=0).T
        # check if shapes match up
        print(features.shape, self.next_price.shape)

        yhat = self.next_price-self.prev_price
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                   # ('linear', Lasso(.0001))])
                   ('linear', LinearRegression(fit_intercept=True))])
        self.model = model.fit(features, yhat) 
        # print(self.model.score(features, yhat))
        self.model = model.fit(features, yhat) 
        print(self.model.score(features, yhat))
        print(self.model.named_steps['poly'].get_feature_names())
        print(self.model.named_steps['linear'].coef_)

    def predict(self, prev_price, x1, x2):
        features = np.stack((x1, x2), axis=0).reshape((1,-1))

        diff = -.1*x1 + .2*x2**2
        new_price = prev_price + diff #self.model.predict(features)[0]
        return new_price

class MediumModel(object):
    def __init__(self):
        self.prev_price, self.x1, self.x2, self.x3, self.next_price = load_medium(train_dir)
        self.train()

    def train(self):
        # train model here
        features = np.stack((self.x1, self.x2, self.x3), axis=0).T
        yhat = np.log(self.next_price)-np.log(self.prev_price)

        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                   # ('linear', Lasso(.1))])
                   ('linear', LinearRegression(fit_intercept=True))])
        self.model = model.fit(features, yhat) 
        print(self.model.score(features, yhat))
        print(self.model.named_steps['poly'].get_feature_names())
        print(self.model.named_steps['linear'].coef_)

    def predict(self, prev_price, x1, x2, x3):
        features = np.stack((x1, x2, x3), axis=0).reshape((1,-1))

        predict = 1e-3 * x2 + 1e-3 * x2**2
        log_new_price = np.log(prev_price) + self.model.predict(features)[0]
        return np.exp(log_new_price)

class HardModel(object):
    def __init__(self):
        self.prev_price, self.x1, self.x2, self.x3, self.next_price = load_hard(train_dir)
        self.train()

    def train(self):
        # train model here
        last_50 = []
        total_points = len(self.prev_price)
        for i in reversed(range(1,51)):
            last_50.append(self.next_price[50-i:-i])

        last_50 = np.array(last_50)
        nx1 = self.x1[50:]
        nx2 = self.x2[50:]
        nx3 = self.x3[50:]
        old_features = np.stack((nx1, nx2, nx3), axis=0)
        print(last_50.shape)
        print(old_features.shape)
        features = np.vstack((last_50, old_features)).T
        print(features.shape)
        yhat = self.next_price[50:]
        # print('next price', self.next_price[0])
        # print('prev price', self.features[0])

        model = Pipeline([('poly', PolynomialFeatures(degree=1)),
                   # ('linear', Lasso(.1))])
                   ('linear', LinearRegression(fit_intercept=True))])
        self.model = model.fit(features, yhat) 
        print(self.model.score(features, yhat))
        print(self.model.named_steps['poly'].get_feature_names())
        # print(self.model.named_steps['linear'].coef_)

    def predict(self, price_history, x1, x2, x3):
        # note price history is the previous 50 prices with most recent prev_price last
        #   and x1, x2, x3 are still single values
        old_features = np.stack((x1, x2, x3), axis=0)
        features = np.concatenate((price_history, old_features), axis=0).reshape((1,-1))
        new_price = self.model.predict(features)[0]
        return new_price

simple_model = SimpleModel()
medium_model = MediumModel()
hard_model = HardModel()

def allocate(simple_args, medium_args, hard_args):
    """
    Implement your allocation function here
    You should return a tuple (a1, a2, a3), where
        a1 is the quantity of stock simple you wish to purchase and so forth
    You will buy the stocks at the current price
    
    The argument format is as follows:
        simple_args will be a tuple of (current_price, current_x1, current_x2)
        medium_args will be a tuple of (current_price, current_x1, current_x2, current_x3)
        hard_args will be a tuple of (current_price_history, current_x1, current_x2, current_x3)
            where the current_price_history is the previous 50 prices
            and the current price is the last element of current_price_history

    Note that although we notate for example feature x1 for all the stocks, the 
        features for each stock are unrelated (x1 for simple has no relation to x1 for medium, etc)

    Make sure the total money you allocate satisfies
        (a1 * simple_current_price + a2 * medium_current_price + a3 * hard_current_price) < 100000000
    Quantities may be decimals so don't worry about rounding
    To be safe, you should make sure you're lower than 100000000 by a threshold
    You can check your code with the provided helper test_allocate.py

    Test your allocation function on the provided test set by running python test_allocate.py
    Generate your final submission on the real data set by running python run_allocate.py
    """
    # Sample: retrieve prices and get predictions from models
    simple_price = simple_args[0]
    medium_price = medium_args[0]
    hard_price = hard_args[0][-1]
    simple_prediction = simple_model.predict(*simple_args)
    medium_prediction = medium_model.predict(*medium_args)
    hard_prediction = hard_model.predict(*hard_args)
    
    simple_delta_pred = simple_prediction - simple_price
    simple_delta_perc_pred = simple_delta_pred/simple_price
    medium_delta_pred = medium_prediction - medium_price
    medium_delta_perc_pred = medium_delta_pred/medium_price
    hard_delta_pred = hard_prediction - hard_price
    hard_delta_perc_pred = hard_delta_pred/hard_price
    
    delta_perc_pred = [simple_delta_perc_pred,medium_delta_perc_pred,hard_delta_perc_pred]
    
    M = 100000000-1
    alpha = 0.0002
    
    all_pos = set()
    if simple_delta_perc_pred > 0:
        all_pos.add(0)
    if medium_delta_perc_pred > 0:
        all_pos.add(1)
    if hard_delta_perc_pred > 0:
        all_pos.add(2)
        
    if len(all_pos) == 0:
        return (0,0,0)
    
    sum_square_deltas = 0
    for i in all_pos:
        sum_square_deltas += delta_perc_pred[i]**2
    
    ret_amts = [0,0,0]
    
    for i in all_pos:
        ret_amts[i] = max((((delta_perc_pred[i]**2)/sum_square_deltas)*(3+alpha*M) - 1)/alpha,0)
    
    ret_amts = np.array(ret_amts)
    total = sum(ret_amts * np.array([simple_price,medium_price,hard_price]))
    if total > M:
        ret_amts *= (M/total)
    return (ret_amts[0],ret_amts[1],ret_amts[2])

# def allocate(simple_args, medium_args, hard_args):
#     """
#     Implement your allocation function here
#     You should return a tuple (a1, a2, a3), where
#         a1 is the quantity of stock simple you wish to purchase and so forth
#     You will buy the stocks at the current price
    
#     The argument format is as follows:
#         simple_args will be a tuple of (current_price, current_x1, current_x2)
#         medium_args will be a tuple of (current_price, current_x1, current_x2, current_x3)
#         hard_args will be a tuple of (current_price_history, current_x1, current_x2, current_x3)
#             where the current_price_history is the previous 50 prices
#             and the current price is the last element of current_price_history

#     Note that although we notate for example feature x1 for all the stocks, the 
#         features for each stock are unrelated (x1 for simple has no relation to x1 for medium, etc)

#     Make sure the total money you allocate satisfies
#         (a1 * simple_current_price + a2 * medium_current_price + a3 * hard_current_price) < 100000000
#     Quantities may be decimals so don't worry about rounding
#     To be safe, you should make sure you're lower than 100000000 by a threshold
#     You can check your code with the provided helper test_allocate.py

#     Test your allocation function on the provided test set by running python test_allocate.py
#     Generate your final submission on the real data set by running python run_allocate.py
#     """
#     # Sample: retrieve prices and get predictions from models
#     simple_price = simple_args[0]
#     medium_price = medium_args[0]
#     hard_price = hard_args[0][-1]
#     simple_prediction = simple_model.predict(*simple_args)
#     medium_prediction = medium_model.predict(*medium_args)
#     hard_prediction = hard_model.predict(*hard_args)

#     # Sample: allocate all money (except a small threshold) to medium
#     if simple_prediction > simple_price:
#         return (0, (100000000 - 1) / medium_price, 0)
#         # return ((100000000 - 1) / simple_price, 0, 0)
#     return (0,0,0)