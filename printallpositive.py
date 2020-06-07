#!/usr/bin/env python
# coding: utf-8

# In[4]:


def mylist(arr):
    update=[]
    for num in arr:
        if num>=0:
            update.append(num)
    return update
        


# In[5]:


result=mylist([12,-7,5,64,-14])
print(result)


# In[ ]:




