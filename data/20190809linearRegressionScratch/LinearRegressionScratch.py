#!/usr/bin/env python
# coding: utf-8

# In[69]:


import matplotlib.pyplot as plt
import random


# In[82]:


x = list(range(-10,12))
x


# In[83]:


y = [2*xval-1 for xval in x]
y


# In[84]:


random.seed(999)
# normalize the values
minx = min(x)
maxx = max(x)
miny = min(y)
maxy = max(y)
x = [(xval - minx)/(maxx-minx) for xval in x]
y = [(yval - miny)/(maxy-miny) + random.random()/5 for yval in y] 
print(x)
print(y)


# In[85]:


plt.plot(x,y,'o')
plt.savefig('fig1.png')


# In[86]:


theta0 = 0.5
theta1 = 0
ypred = [theta0 + theta1*xval for xval in x]

plt.plot(x,y,'o')
plt.plot(x,ypred,'+')
plt.savefig('fig2.png')


# In[87]:


epochs = 100 # number of iterations
learning_rate = 0.05

theta_history = [[theta0,theta1]]
J = list()
for epoch in range(epochs):
    J.append((sum([(ypredval-yval)**2 for ypredval,yval in zip(ypred,y)])))
    print('J = ',J[-1])
    
    dJd0 = (sum([ypredval - yval for ypredval,yval in zip(ypred,y)]))
    dJd1 = (sum([(ypredval - yval)*xval for ypredval,yval,xval in zip(ypred,y,x)]))
    
    theta0 = theta0 - learning_rate*dJd0
    theta1 = theta1 - learning_rate*dJd1
    
    theta_history.append([theta0,theta1])
    
    ypred = [theta0 + theta1*xval for xval in x]


# In[88]:


plt.plot(J)
print(theta0,theta1)
plt.savefig('fig3.png')


# In[89]:


plt.plot(x,y,'o')
plt.plot(x,ypred,'+')
plt.savefig('fig4.png')


# In[93]:


# check convergence and stop (early stopping)
epochs = 100 # number of iterations
learning_rate = 0.05

theta0 = 0.5
theta1 = 0
theta_history = [[theta0,theta1]]
J = list()
for epoch in range(epochs):
    J.append((sum([(ypredval-yval)**2 for ypredval,yval in zip(ypred,y)])))
    print('J = ',J[-1])
    
    dJd0 = (sum([ypredval - yval for ypredval,yval in zip(ypred,y)]))
    dJd1 = (sum([(ypredval - yval)*xval for ypredval,yval,xval in zip(ypred,y,x)]))
    
    theta0 = theta0 - learning_rate*dJd0
    theta1 = theta1 - learning_rate*dJd1
    
    theta_history.append([theta0,theta1])
    
    ypred = [theta0 + theta1*xval for xval in x]
    
    if abs(learning_rate*dJd0) < 1e-3 and abs(learning_rate*dJd1) < 1e-3:
        break


# In[94]:


plt.plot(J)
print(theta0,theta1)
plt.savefig('fig5.png')


# In[95]:


plt.plot(x,y,'o')
plt.plot(x,ypred,'+')
plt.savefig('fig6.png')


# In[ ]:




