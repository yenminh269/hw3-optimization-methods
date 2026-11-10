import numpy as np

#benchmark test  functions
def sphere(theta):
  x, y = theta
  return x**2 + y**2

def sphere_gradient(theta):
  x, y = theta

  return np.array([2 * x, 2 * y])

def rosenbrock(theta):
  beta, gamma = theta

  return 100 * (gamma - beta**2)**2 + (1 - beta)**2

def rosenbrock_gradient(theta):
  beta, gamma = theta

  return np.array([ 
    -400 * beta * (gamma - beta**2) 
    - 2 * (1 - beta), 
    200 * (gamma - beta**2)               
  ])

