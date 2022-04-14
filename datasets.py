import numpy as np
import cv2
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import os


class MiniImagenetDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        super(MiniImagenetDataset, self).__init__()
        self.root = root
        self.modes = os.listdir(self.root)
        self.transform = transform
        Images, Y = [], []

        gt_list=[]
        for i in range(len(self.modes)): gt_list += os.listdir(self.root+self.modes[i]) # list of all labels
         
        for i in range(len(self.modes)):
            mode_root = self.root + self.modes[i] + '/'
            folders = os.listdir(mode_root)

            folders = os.listdir(mode_root)
            for folder in folders:
                folder_path = os.path.join(mode_root, folder)
                for ims in os.listdir(folder_path):
                    try:
                        img_path = os.path.join(folder_path, ims)
                        Images.append(np.array(cv2.imread(img_path)))
                        Y.append(gt_list.index(folder))  # return index of the gt
                    except:
                        # Some images in the dataset are damaged
                        print("File {}/{} is broken".format(folder, ims))
        data = [(x, y) for x, y in zip(Images, Y)]

        train_data, test_data = self.train_test_split(data, 500, 100)       
        if mode == 'test':
            self.data = test_data
        if mode == 'train':
            self.data = train_data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img = self.data[index][0]
        try:
            img = Image.fromarray((img).astype(np.uint8))
            label = self.data[index][1]
        except:
            img = self.data[index-1][0]                           # stupid Nonetype handling
            img = Image.fromarray((img).astype(np.uint8))
            label = self.data[index-1][1]

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

    def train_test_split(self, data, n_train_per_class, n_test_per_class):
        train_data, test_data = [], []
        n_iter = len(data)//(n_test_per_class+n_train_per_class)
        c = 0
        for _ in range(n_iter):
            for _ in range(n_train_per_class):
                train_data.append(data[c])
                c+=1
            for _ in range(n_test_per_class):
                test_data.append(data[c])
                c+=1

        return train_data, test_data

