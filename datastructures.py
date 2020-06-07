#!/usr/bin/env python
# coding: utf-8

# lists

# In[1]:


list1=[1,2,3,4,"pineapple","grapes","mango",33.33,3.14,0.000000067]


# In[2]:


list1[0]="coconut"


# In[3]:


print(list1)


# tuple

# In[4]:


t1=(1,2,3,4,5,6,7,8,9)


# In[5]:


t1[0]


# In[6]:


t1[5]


# In[7]:


t1[3]=55
#it shows error because tuples are unmutable

# dictionary

# In[8]:


my_dict = {'key1':'value1','key2':'value2','key3':'value3','key4':'value4'}


# In[9]:


del my_dict['key4']


# In[10]:


print(my_dict)


# In[11]:


deleted=my_dict.pop('key1')


# In[12]:


print(deleted)


# In[13]:


print(my_dict)

