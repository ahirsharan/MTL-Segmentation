
""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

#For COCO
class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args, train_aug=False):
        self.args=args
        # Set the path according to train, val and test        
        if setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'train/')
            THE_PATHL = osp.join(args.dataset_dir, 'labels/train/')
        elif setname=='val':
            THE_PATH = osp.join(args.dataset_dir, 'val/')
            THE_PATHL = osp.join(args.dataset_dir, 'labels/val/')
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label       
        
        # exit()    
        data = []
        label = []

        # Get folders' name
        #folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]
        
        # Get the images' paths and labels
        input_images = os.listdir(THE_PATH)      
        label_images = os.listdir(THE_PATHL)
        
        #for idx, this_folder in enumerate(folders):
            #this_folder_images = os.listdir(this_folder)
            # print(idx)
            # exit()
        for labimage_path in label_images:
            data.append(osp.join(THE_PATH, labimage_path[:-3]+'jpg'))
            label.append(osp.join(THE_PATHL, labimage_path))

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        
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
        num_classes=self.args.num_classes
        inppath, labpath = self.data[i], self.label[i]
        inpimage = self.transform(Image.open(inppath).convert('RGB'))
        labimage = self.btransform(Image.open(labpath).convert('LA'))
        labimage=(num_classes-1)*labimage
        labimage=labimage.long()
        return inpimage,labimage[0]
