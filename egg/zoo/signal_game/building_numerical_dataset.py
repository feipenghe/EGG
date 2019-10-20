#!/usr/bin/env python
# coding: utf-8
class SignalGameDataset:
    def __init__(self, data):
        # data input was context and now it is changed into numpy or regular array
        self.data = data
        self.num_samples = 2
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def getloader(self, batch_size, shuffle = True):
        sg_dataset = SignalGameDataset(self.data)
        import torch
        from torch.utils.data import DataLoader 
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle)


