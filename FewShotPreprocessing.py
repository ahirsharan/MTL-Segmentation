
import os
import os.path as osp
from PIL import Image
    
PATH='../Fewshot/Fewshot/'
classes= os.listdir(PATH)
trainp='../Fewshot/train/'
valp='../Fewshot/val/'
testp='../Fewshot/test/'

for classv in classes:
    if classv[0]=='.':
        continue
    pathn=osp.join(PATH,classv)
    pathn=pathn+'/'
    folders=os.listdir(pathn)
    
    path1=osp.join(trainp,'images/')
    path1=osp.join(path1,classv)
    os.mkdir(path1)
    path1 =path1 +'/'
    
    path2=osp.join(trainp,'labels/')
    path2=osp.join(path2,classv)
    os.mkdir(path2)
    path2=path2+'/'   
    
    
    for i in range(0,8,1):
        p=osp.join(pathn,folders[i])
        im=Image.open(p)
        if(i%2==0):
            p1=osp.join(path1,folders[i])
            im.save(p1)
        else:
            p2=osp.join(path2,folders[i])
            im.save(p2)
            
    path1=osp.join(valp,'images/')
    path1=osp.join(path1,classv)
    os.mkdir(path1)
    path1 =path1 +'/'
    
    path2=osp.join(valp,'labels/')
    path2=osp.join(path2,classv)
    os.mkdir(path2)
    path2=path2+'/'   
    
    
    for i in range(8,16,1):
        p=osp.join(pathn,folders[i])
        im=Image.open(p)
        if(i%2==0):
            p1=osp.join(path1,folders[i])
            im.save(p1)
        else:
            p2=osp.join(path2,folders[i])
            im.save(p2)            
            
            
    path1=osp.join(testp,'images/')
    path1=osp.join(path1,classv)
    os.mkdir(path1)
    path1=path1+'/'
    
    path2=osp.join(testp,'labels/')
    path2=osp.join(path2,classv)
    os.mkdir(path2)            
    path2=path2+'/'            
            
    for i in range(16,20,1):
        p=osp.join(pathn,folders[i])
        im=Image.open(p)
        if(i%2==0):
            p1=osp.join(path1,folders[i])
            im.save(p1)
        else:
            p2=osp.join(path2,folders[i])
            im.save(p2)      
