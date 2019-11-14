#!/usr/bin/env python
# coding: utf-8
<<<<<<< HEAD
from torch.utils.data import Dataset
import torch
class SignalGameDataset(Dataset):
    # def __init__(self, data):
    def __init__(self):
        # data input was context and now it is changed into numpy or regular array
        # self.data = data
        self.num_samples = 2
        self.samples = list(range(0, 32))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        n = self.samples[index]  # where do the samples come from?
        successors = torch.arange(32).float() + n + 1
        noisy = torch.randn(32) + successors
        return n, successors, noisy
    
    def getloader(self, batch_size, shuffle = True):
        # sg_dataset = SignalGameDataset(self.data)
        sg_dataset = SignalGameDataset()
        from torch.utils.data import DataLoader
        # print("dataset dim: " + str(sg_dataset.shape))
        # return sg_dataset
        data_loader = DataLoader(sg_dataset, batch_size = batch_size, shuffle = shuffle)
        print(next(iter(data_loader)))
        return data_loader
        # dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)
=======
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
>>>>>>> bb771d9d5754b6b75116676292d59dadb1dbb13e




class SignalGameLoader(torch.utils.data.DataLoader):
    def __init__(self,n_features, batches_per_epoch, ):