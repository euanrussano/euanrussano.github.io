#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.optimize

# Problem variables
F = 1.0*10**4    # kg-water / h
x0 = 0.02       # kg-solute / kg-water
s = 7.0*10**-4   # kg-solvent / kg-water
m = 4.0         # kg-water / kg solvent
Ps = 0.4        # USD / kg-solute
Px = 0.01       # USD / kg-solvent.


# In[2]:


def problem(x):

    W1 = x[0] # mass flow rate
    W2 = x[1] # mass flow rate
    W_1 = x[2] # mass flow rate
    W_2 = x[3] # mass flow rate
    x1 = x[4]  # liquid molar composition
    y1 = x[5]  # gas molar composition
    x2 = x[6]  # liquid molar composition
    y2 = x[7]  # gas molar composition

    # Income
    R = Ps*(W_1*y1+W_2*y2)
    
    # Cost
    C = Px*(W1+W2)
    
    # Profit (negative for minimization)
    L = -(R-C)
    
    return L


# In[3]:


def cons(x):

    W1 = x[0]
    W2 = x[1]
    W_1 = x[2]
    W_2 = x[3]
    x1 = x[4]
    y1 = x[5]
    x2 = x[6]
    y2 = x[7]

    cons = np.zeros(6)

    # Solute mass balance
    cons[0] = F*x0-W_1*y1-F*x1
    cons[1] = F*x1-W_2*y2-F*x2

    # Solvent mass balance
    cons[2] = W1-W_1-s*F
    cons[3] = W2+s*F-W_2-s*F

    # Equilibrium relations
    cons[4] = y1-m*x1
    cons[5] = y2-m*x2

    return cons


# In[5]:


xi = np.zeros(8)
x = scipy.optimize.minimize(problem, xi, constraints={'type':'eq','fun':cons})


print('Optimization Result \n')
print('W1 = {:.3f}'.format(x.x[0]))
print('W2 = {:.3f}'.format(x.x[1]))
print('W_1 = {:.3f}'.format(x.x[2]))
print('W_2 = {:.3f}'.format(x.x[3]))
print('x1 = {:.3f}'.format(x.x[4]))
print('y1 = {:.3f}'.format(x.x[5]))
print('x2 = {:.3f}'.format(x.x[6]))
print('y2 = {:.3f}'.format(x.x[7]))

