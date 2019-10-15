#!/usr/bin/env python
# coding: utf-8
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


