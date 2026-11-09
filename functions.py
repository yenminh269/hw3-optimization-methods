import numpy as np

#verification functions
def sphere(theta):
  x1, x2 = theta
  return x1**2 + x2**2

def sphere_gradient(theta):
  x1, x2 = theta

  return np.array([2 * x1, 2 * x2])

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

