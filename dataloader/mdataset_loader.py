
""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

#For FewShot
class mDatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, train_aug=False):
        # Set the path according to train, val and test
        self.args=args
        if setname=='meta':
            THE_PATH = osp.join(args.mdataset_dir, 'train/images')
            THE_PATHL = osp.join(args.mdataset_dir, 'train/labels/')
        elif setname=='val':
            THE_PATH = osp.join(args.mdataset_dir, 'val/images/')
            THE_PATHL = osp.join(args.mdataset_dir, 'val/labels/')
        elif setname=='test':
            THE_PATH = osp.join(args.mdataset_dir, 'test/images/')
            THE_PATHL = osp.join(args.mdataset_dir, 'test/labels/')            
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label       
        
        # exit()    
        data = []
        label = []
        labeln=[]
        
        # Get the classes' names
        folders = os.listdir(THE_PATH)      
        
        for idx, this_folder in enumerate(folders):
            
            imf=osp.join(THE_PATH,this_folder)
            imf=imf+'/'
            this_folder_images = os.listdir(imf)
            for im in this_folder_images:
                data.append(osp.join(imf, im))
            
            lbf=osp.join(THE_PATHL,this_folder)
            lbf=lbf+'/'
            this_folder_images = os.listdir(lbf)
            for lb in this_folder_images:
                label.append(osp.join(lbf, lb))    
                labeln.append(idx)
            
        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.labeln=labeln
        
        # Transformation for RGB
        if train_aug:
            image_size = 284
            self.transform = transforms.Compose([
                transforms.Resize(290),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()])
                #transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            image_size = 284
            self.transform = transforms.Compose([
                transforms.Resize(290),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()])
                #transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])


        # Transformation for label BW
        if train_aug:
            image_size = 284
            self.btransform = transforms.Compose([
                transforms.Resize(290),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()])
                #transforms.Normalize(np.array([x / 255.0 for x in [125.3]]), np.array([x / 255.0 for x in [63.0]]))])
        else:
            image_size = 284
            self.btransform = transforms.Compose([
                transforms.Resize(290),
                transforms.CenterCrop(image_size),
                transforms.ToTensor()])
                #transforms.Normalize(np.array([x / 255.0 for x in [125.3]]),np.array([x / 255.0 for x in [63.0]]))])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        inppath, labpath,idx = self.data[i], self.label[i],self.labeln[i]
        inpimage = self.transform(Image.open(inppath).convert('RGB'))
        labimage = self.btransform(Image.open(labpath).convert('LA'))
        labimage=(self.args.way-1)*labimage
        labimage=labimage.long()
        return inpimage,labimage[0],idx
