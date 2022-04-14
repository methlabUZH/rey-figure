#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 


# In[45]:


IMAGE_SIZE = "232 x 300"
SAVE_DIR = "./ablation_study_{}".format(IMAGE_SIZE)


# In[43]:


results = [1.228, 1.1613, 1.259, 1.148] 
labels = ['Vanilla', 'Only data aug', 'Only tta', 'Both tta + data aug']


# In[46]:


fig = plt.figure(figsize = (10, 5))
plt.rcParams['savefig.facecolor']='white'
# creating the bar plot
plt.bar(labels, results, color ='maroon', width = 0.5)
# set y value range of plot
plt.ylim(1.1, 1.3)
plt.ylabel("MAE")
plt.title(f"Ablation study of different training and inference techniques (IMAGE_SIZE = {IMAGE_SIZE})")
plt.axhline(y=1.148,linewidth=1, color='k', linestyle='--')
plt.savefig("ablation_study.png")
plt.show()


# In[ ]:




