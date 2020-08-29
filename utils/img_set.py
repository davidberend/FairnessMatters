from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

class Img_Dataset(Dataset):
    '''
     imgs: numpy array, 0-255
     img_size: width*height
    '''
    def __init__(self, imgs ,img_size, labels = None, transform = None):
        
        self.transform = transform
        self.labels = labels
        self.total_num = imgs.shape[0]
        def load_image(idx):
            cur_img_arr = imgs[idx]
            img = Image.fromarray(np.uint8(cur_img_arr))
            return img.resize(img_size)
        self.load_image = load_image
    
    def __getitem__(self,index):
        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            label = 0
        else:
            label = self.labels[index]
        return img, label  # 0 is the class
    
    def __len__(self):
        return self.total_num


class Img_Dataset_Iter(Dataset):
    '''
     imgs: numpy array, 0-255
     img_size: width*height
    '''
    def __init__(self, img_paths ,img_size, labels = None, transform = None):
        
        self.transform = transform
        self.labels = labels
        self.total_num = len(img_paths)
        def load_image(idx):
            cur_img_arr = img_paths[idx]
            img = Image.open(cur_img_arr)
            if img.mode=='L':
              img=img.convert("RGB")
            newimg=img.resize(img_size)
            img.load()
            return newimg
        self.load_image = load_image
    
    def __getitem__(self,index):
        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            label = 0
        else:
            label = int(self.labels[index])
        return img, label  # 0 is the class
    
    def __len__(self):
        return self.total_num