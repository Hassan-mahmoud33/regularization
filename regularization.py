import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

def delete():
    os.system('cls' if os.name == 'nt' else 'clear')
delete()

# load dataset
data = pd.read_csv("ex2data2.txt" , header=None , names=['Test 1' , 'Test 2' , 'Accepted'] , skipinitialspace=True)

def display():
    print(f'{'-'*50}\n')
    print(data.head(3))
    print(f'{'-'*50}\n')

    print(data.describe())
    print(f'{'-'*50}\n')
    print(data.info())
    print(data.nunique())

# display()



positive = data [ data['Accepted'].isin([1])]
negative = data [ data['Accepted'].isin([0])]

# print(positive.head(5))
# print(f"{'-'*50}\n")
# print(negative.head(5))


# Visualize data

fig , ax = plt.subplots(figsize = ( 7 , 5))
ax.scatter( positive['Test 1'] , positive['Test 2'], marker = 'o', c= 'g', s = 50 , label = 'Accepted')

ax.scatter( negative['Test 1'] , negative['Test 2'], marker = 'x', c = 'r', s = 50 , label = 'Rejected')
ax.legend(loc = 1)
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2')
ax.set_title('Test 1 Score group vs Test 2 Score group')
plt.show()




x1 = data['Test 1']
x2 = data['Test 2']

# print(x1.head(5))
# print(f"{'-'*50}\n")
# print(x2.head(5))
# print(f"{'-'*50}\n")


data.insert( 3 , 'ones' , 1)
# print(data.head(5))


# Feature Mapping
degree = 6 

for i in range(1 , degree + 1   ):
    for j in range( 0 , i):
        data[f"F{i}{j}"] =  np.power(x1 , i - j ) * np.power( x2 , j )

# print(data.head(2))


data.drop('Test 1' , axis=1  ,inplace=True) 
data.drop('Test 2' , axis=1  ,inplace=True) 

# print(data.head(2))




cols = data.shape[1]
X = data.iloc[ : , 1 : cols ]
y = data.iloc[ : , 0 : 1]

# print(X.head(2))
# print(y.head(2))

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(X.shape[1])

a = 0.001

# print(X[ : 3])
# print('-'*80)
# print(y[ : 3])
# print(theta)



def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))


def cost(theta , X , y , a):

    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply( -y , np.log( sigmoid( X * theta.T  )) )
    second = np.multiply( ( 1 - y ) , np.log( 1 - sigmoid( X * theta.T  )))
    reg = ( a / ( 2 * len(X)) ) * ( np.sum( np.power(theta[ : , 1 : theta.shape[1]] , 2 )))  

    return np.sum( first - second ) / (len(X)) + reg



def gradient( theta ,  X , y , a):

    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int (theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid( X * theta.T ) - y

    for i in range(parameters):
        term = np.multiply( error , X[ : , i ])

        if i == 0:
            grad[i]= np.sum(term) / len(X)
        
        else :
           
            grad[i]= np.sum(term) / len(X) + ( ( a / len(X)) *  theta[ : ,  i].item() )

    return grad



rcost = cost( theta , X , y , a)

# Optimization
# library that give you the lower (minimum) value for a math function
import scipy.optimize as opt

result = opt.fmin_tnc ( func=cost , x0=theta , fprime=gradient , args=( X , y , a) , messages=False)

print(f"result = {result}") 


cost_after_opt = cost( result[0] , X , y , a)

print(f"\ncost before optimization = {rcost}")
print(f"cost after optimization = {cost_after_opt}\n")



# prediction

def predict( theta , X):

    probability = sigmoid( X * theta.T )
    return [ 1 if x >= 0.50 else 0 for x in probability ]

theta_min = np.matrix(result[0])
predictions = predict(theta_min , X )

print(f"new predictions = {predictions[ : 30]}\n")

correct = [ 1 if ( a == b ) else 0  for  a , b  in zip( predictions , y )]


accuracy = ( sum(map( int , correct)) / len(correct) * 100 )
print(f"Accuracy = {accuracy} %")

