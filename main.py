# importing the libraries        
import scipy as sp
import matplotlib.pyplot as plt
import os
import random
from scipy import stats


# setting the directory for data and charts

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "Data")

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")

# creating directories if it does not exist
for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)


# properties of grapgh
colors = ['g', 'b', 'm', 'r', 'darkolivegreen', 'c']
linestyles = ['-', '-.', '--', ':', '-', '-']

       

def get_data():
    # data is a 2d array which now contains the data
    data = sp.genfromtxt(os.path.join(DATA_DIR, "visitors_per_hour.txt"), delimiter="\t")
    
    temp = data[0:740*24, 1]
    temp_1 = data[0:740*24, 0]
    
    day = sp.arange(1, 366, 1)
    #print(day)
    hits = data[0:731, 1]
    
    #print(temp)
    
    # Cleaning the data i.e. removing NAN values if present
    temp = temp[~sp.isnan(temp)]

        
    
    for i in range(1, len(temp)):
        temp_1[i-1] = temp[i] - temp[i-1]   
    
    #print(temp_1)
    
    for i in range(0,731):
        x = 0
        for j in range(0,24):
            x += temp_1[i*24 + j]
        hits[i] = x
    
    
    sp.savetxt(os.path.join(DATA_DIR, "visitors_per_day_train.txt"), sp.column_stack([day.astype(int), hits[0:365].astype(int)]), fmt='%.18g' , delimiter=' ')

    return [day, hits]

# end of get_data()

   
def treat_outliers(hits):
    Mean = stats.describe(hits).mean
    Standard_deviation = int(sp.sqrt(stats.describe(hits).variance))
    # Choosing x = 1 and y = 2 for the Sigma approach  
    # Upper limit = Mean + x*Standard_deviation
    # Lower limit = Mean - x*Standard_deviation
    x = 1
    y = 2
    UL = int(Mean + x*Standard_deviation)
    LL = int(Mean - x*Standard_deviation)
    #print(UL)
    #print(LL)
    for i in range(0, len(hits)):
        if hits[i] > UL :
            hits[i] = hits[i] - y*Standard_deviation
        elif hits[i] < LL : 
            hits[i] = hits[i] + y*Standard_deviation   
    
    cap = sp.percentile(hits, 99)
    #print(cap)
    for i in range(0, len(hits)):
        if hits[i] >= cap:
            hits[i] = 1444
    return hits    
    

def compute_error(day, hits, c, m):
    total_err = 0  
    for i in range(len(day)):
        x = float(day[i])
        y = float(hits[i])
        #print(f"{i} {x} {y}")
        total_err += (y - (m * x + c )) ** 2
    
    return total_err / float(len(day))

def step_gradient(c_current, m_current, day, hits, learning_rate):
    c_gradient = 0
    m_gradient = 0
    N = len(day)
        
    for i in range(len(day)):
        x = float(day[i])
        y = float(hits[i])
        c_gradient += -(2/N) * (y - ((m_current * x) + c_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + c_current))
        
        
    new_c = c_current - (learning_rate * c_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_c, new_m]



def grad_descent(day, hits, starting_c, starting_m, learning_rate, iterations):
    c = starting_c
    m = starting_m
    for i in range(iterations):
        c, m = step_gradient(c, m, day, hits, learning_rate)
        #print (f"{b}    {m}")
    return [c, m]

    
                
# plot graphs       
def plot_graph(a, b, models, name, mx=None, ymax=3000, xmin=None):
    # define the size of the figure
    plt.figure(figsize=(8, 6))
    
    # clear the current figure
    plt.clf()
    
    # plot the visitors data
    plt.scatter(a, b, s=5)
    
    # title of the figure
    plt.title("Web traffic over a Year")
    
    # defining x and y axes labels
    plt.xlabel("Day")
    plt.ylabel("Hits/day")    
    # plot the model
    if models:
        if mx is None:
            # create a vector of 1000 equally spaced integers in the 
            # range 0 to x[-1] i.e. the last data point
            # values are used for plotting on x-axis
            mx = sp.linspace(0, day[-1], 1000)
            
        # the zip function creates a tuple which have the values 
        # (model, linestyle, color) on which the for loop is used
        for model, style, color in zip(models, linestyles, colors):
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)
        
        plt.legend(["d=%i" % m.order for m in models], loc="upper left")
        
    plt.autoscale(tight=True)
    
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
        
    # draw a light grey colored grid     
    plt.grid(True, linestyle='-', color='0.8')
    
    # save the graph in the CHART_DIR
    plt.savefig(name)
    
    # print the graph on the output
    plt.show()
    
     

# main 
[day,hits] = get_data()

# The data which will be used to train our model
train_hits = hits[0:365]

# The data which will be used to test our model
test_data = sp.genfromtxt(os.path.join(DATA_DIR, "visitors_per_day_test.txt"), delimiter=" ")
test_hits = test_data[0:365, 1]
print(test_hits)

#plot the initial data
plot_graph(day, train_hits, None, os.path.join(CHART_DIR, "data_initial.png"))


#Removing Outliers to improve efficiency
train_hits = treat_outliers(train_hits)
plot_graph(day, train_hits, None, os.path.join(CHART_DIR, "data_clean.png"), None, 2000)



fp1, res1, rank1, sv1, rcond1 = sp.polyfit(day, train_hits, 1, full=True)
print("Model parameters of fp1: %s" % fp1)
print("Error of the model of fp1:", res1)
f1 = sp.poly1d(fp1)
print("The equation is: %s" % f1)
plot_graph(day, train_hits, [f1], os.path.join(CHART_DIR, "data_1.png"), None, 2000)


#Fitting a 2 degree model
fp2, res2, rank2, sv2, rcond2 = sp.polyfit(day, train_hits, 2, full=True)
print("Model parameters of fp2: %s" % fp2)
print("Error of the model of fp2:", res2)
f2 = sp.poly1d(fp2)
plot_graph(day, train_hits, [f1, f2], os.path.join(CHART_DIR, "data_2.png"), None, 2000)


#Fitting a 3 degree model
fp3, res3, rank3, sv3, rcond3 = sp.polyfit(day, train_hits, 3, full=True)
print("Model parameters of fp3: %s" % fp3)
print("Error of the model of fp3:", res3)
f3 = sp.poly1d(fp3)
plot_graph(day, train_hits, [f1, f2, f3], os.path.join(CHART_DIR, "data_3.png"), None, 2000)


#Fitting a 4 degree model
fp4, res4, rank4, sv4, rcond4 = sp.polyfit(day, train_hits, 4, full=True)
print("Model parameters of fp4: %s" % fp4)
print("Error of the model of fp4:", res4)
f4 = sp.poly1d(fp4)
plot_graph(day, train_hits, [f1, f2, f3, f4], os.path.join(CHART_DIR, "data_4.png"), None, 2000)


#Fitting a 10 degree model
fp10, res10, rank10, sv10, rcond10 = sp.polyfit(day, train_hits, 10, full=True)
print("Model parameters of fp10: %s" % fp10)
print("Error of the model of fp10:", res10)
f10 = sp.poly1d(fp10)
plot_graph(day, train_hits, [f1, f2, f3, f4, f10], os.path.join(CHART_DIR, "data_10.png"), None, 2000)

fp100 = sp.polyfit(day, train_hits, 100)
f100 = sp.poly1d(fp100)
plot_graph(day, train_hits, [f1, f2, f3, f4, f10, f100], os.path.join(CHART_DIR, "data_100.png"), None, 2000)


#Applying these models on the test data

#defining the Error Funtion
def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


#treating outliers
test_hits = treat_outliers(test_hits)

#plotting the cleaned test data
plot_graph(day, test_hits, None, os.path.join(CHART_DIR, "test_clean.png"), None, 2000)

plot_graph(day, test_hits, [f1, f2, f3, f4, f10, f100], os.path.join(CHART_DIR, "test_100.png"), None, 2000)

for f in [f1, f2, f3, f4, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, day, test_hits)))


dp = test_hits
for i in range(0, 365):
   dp[i] = fp4[0]*pow(i+1,4) + fp4[1]*pow(i+1,3) + fp4[2]*pow(i+1,2) + fp4[3]*pow(i+1,1) + fp4[4] 


plot_graph(day, dp.astype(int), None, os.path.join(CHART_DIR, "prediction.png"), None, 2000)

f100 = sp.poly1d(fp100)


#day = day[0:150]
#hits = hits[0:150]


#learning_rate = 0.0001
#iterations = 1000

# assuming equation is y = mx + c
# initial bias or y-intercept
#initial_c = 0

# initial slope
#initial_m = 0

#print(day)
#print(hits)
#plot_graph(day, hits, None, os.path.join(CHART_DIR, "man1.png"))
#Using gradient descent for Linear Regression
#print(f"Starting gradient descent at c = {initial_c}, m = {initial_m} and Error = {compute_error(day,hits,initial_c,initial_m)}")
#print("Computing....")

#[c,m] = grad_descent(day, hits, initial_c, initial_m, learning_rate, iterations)
#print(c)
#f1 = sp.poly1d([c,m])
#print(f1)
#print(fp1)
#plot_graph(day, hits, [f1], os.path.join(CHART_DIR, "demo2.png"))
#print(f"After {iterations} iterations c = {c} and m = {m} and Error = {compute_error(day, hits, c,m)}")