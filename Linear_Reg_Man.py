# importing the libraries        
import scipy as sp
import matplotlib.pyplot as plt
import os
#from decimal import Decimal as dc


# setting the directory for data and charts

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")

# creating directories if it does not exist
for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)
 

       

def get_data():
    # data is a 2d array which now contains the data
    data = sp.genfromtxt(os.path.join(DATA_DIR, "web_traffic.tsv"), delimiter="\t")
    
    # day is the array that contains the sequence of days
    day = data[:160, 0]
    
    # y is the array that contains the number of visitors corresponding to the 
    # particular day
    hits = data[:160, 1]
    
    # Cleaning the data i.e. removing NAN values if present
    day = day[~sp.isnan(hits)]
    hits = hits[~sp.isnan(hits)]
    #print(day)
    
    return [day ,hits]

# end of get_data()
    
def compute_error(day, hits, b, m):
    total_err = 0
h    
    for i in range(len(day)):
        x = float(day[i])
        y = float(hits[i])
        #print(f"{i} {x} {y}")
        total_err += (y - (m * x + b )) ** 2
    
    return total_err / float(len(day))

def step_gradient(b_current, m_current, day, hits, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = len(day)
    for i in range(len(day)):
        x = float(day[i])
        y = float(hits[i])
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def grad_descent(day, hits, starting_b, starting_m, learning_rate, iterations):
    b = starting_b
    m = starting_m
    for i in range(iterations):
        b, m = step_gradient(b, m, day, hits, learning_rate)
        #print (f"{b}    {m}")
    return [b, m]


# main 
[day,hits] = get_data()
learning_rate = 0.0001
iterations = 10000


# assuming equation is y = mx + b
# initial bias or y-intercept
initial_b = 0

# initial slope
initial_m = 0

# Using gradient descent for Linear Regression
print(f"Starting gradient descent at b = {initial_b}, m = {initial_m} and Error = {compute_error(day,hits,initial_b,initial_m)}")
print("Computing....")

[b,m] = grad_descent(day, hits, initial_b, initial_m, learning_rate, iterations)
print(f"After {iterations} iterations b = {b} and m = {m} and Error = {compute_error(day, hits, b,m)}")




