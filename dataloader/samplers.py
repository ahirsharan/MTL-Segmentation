
""" Sampler for dataloader. """
import torch
import numpy as np

# Customize such as total way number of distint classes to segment in a meta task

class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, labeln, n_batch, n_cls, n_per,n_shot):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.n_shot = n_shot

        labeln = np.array(labeln)
        self.m_ind = []
        for i in range(max(labeln) + 1):
            ind = np.argwhere(labeln == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            lr=[]
            dr=[]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                m=l[pos]
                
                for i in range(0,self.n_shot):
                    lr.append(m[i])
                    
                for i in range(self.n_shot,self.n_per):
                    dr.append(m[i])
                    
            batch=[]
            for i in range(len(lr)):
                batch.append(lr[i])
            
            for i in range(len(dr)):
                batch.append(dr[i])
                
            batch = torch.stack(batch).t().reshape(-1)  
            yield batch
