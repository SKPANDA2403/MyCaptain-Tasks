#!/usr/bin/env python
# coding: utf-8

# In[3]:


num= int(input("Enter the number of elements of fibonacci series: "))
arr=[0,1]
a=0
b=1
if num<1:
    print("enter a valid input")
else:
    for i in range(2,num):
        c=a+b
        arr.append(c)
        a=b
        b=c
print(arr)


# In[ ]:




