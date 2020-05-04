import numpy as np
import os
import h5py
from PIL import Image

def load_data(data_dir):
    data=[]
    labels=[]
    img_folders=os.listdir(data_dir)
    try:
        img_folders.remove('.DS_Store')
    except:
        pass

    count=0
    for folder in img_folders:
        path=data_dir+'/'+folder
        imgs_list=os.listdir(path)
        label=int(folder.split('_')[0])
        try:
            imgs_list.remove('.DS_Store')
        except:
            pass
        for img_name in imgs_list:
            img=Image.open(path+'/'+img_name)
            img=np.array(img)
            img=img[:,:,:3]
            img=np.moveaxis(img,-1,0)
            data.append(img)
            labels.append(label)
        count+=1
        if count%50==0:
            print(str(count)+' of '+str(len(img_folders)))

    data=np.array(data)
    labels=np.array(labels)
    return (data,labels)

x,y=load_data('./data/train')
print(x.shape,y.shape)
with h5py.File('train_data.h5',"w") as out:
    out.create_dataset("x",data=x)
    out.create_dataset("y",data=y)

x,y=load_data('./data/val')
print(x.shape,y.shape)
with h5py.File('val_data.h5',"w") as out:
    out.create_dataset("x",data=x)
    out.create_dataset("y",data=y)

x,y=load_data('./data/test')
print(x.shape,y.shape)
with h5py.File('test_data.h5',"w") as out:
    out.create_dataset("x",data=x)
    out.create_dataset("y",data=y)
