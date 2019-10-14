#!/usr/bin/env python
# coding: utf-8

# In[23]:





# In[24]:





# In[25]:


class SignalGameDataset:
    def __init__(self, data):
        self.c = data
        
    def __len__(self):
        return len(self.c.dims)
    
    def __getitem__(self, n):
        return self.c.dims[n]
    
    def getloader(self, batch_size, shuffle = True):
        sg_dataset = SignalGameDataset(self.c)
        import torch
        from torch.utils.data import DataLoader 
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle)
        


# In[26]:


# sg_dataset = SignalGameDataset(c.dims)
# import torch
# from torch.utils.data import DataLoader 
# sg_dataloader = sg_dataset.getloader(32)


# In[27]:


# sg_dataset = SignalGameDataset(c.dims)
# import torch
# from torch.utils.data import DataLoader 
# sg_dataloader = DataLoader(sg_dataset, batch_size = 10, shuffle = True)


# # In[28]:


# iterator = iter(sg_dataloader)

# # while iterator.next():
# #     print(next(sg_dataloader))

# print(next(iter(sg_dataloader)))

