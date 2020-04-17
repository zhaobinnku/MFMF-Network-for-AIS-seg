import torch
import torch.utils.data as data
import os.path
import numpy as np
from create_dataframe import create_df
from augmentations import rotate, HorizontallyFlip, VerticallyFlip

class train_strokelesion_Dataset(data.Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.dataframe = create_df(img_dir)
        self.transform = transform
        self.dataframe.to_csv('df_train.csv',sep='\t')

    def __getitem__(self, index):

        label = self.dataframe.iloc[index,1]
        img = np.load(os.path.join(self.img_dir,self.dataframe.iloc[index,0]))
        if self.transform:
            if np.random.rand() < 0.5:
                if np.random.rand() >0.5:
                    img = HorizontallyFlip(img)
            if np.random.rand() < 0.5:
                if np.random.rand() >0.5:
                    img = VerticallyFlip(img)
            if np.random.rand() < 0.5:
                if np.random.rand()>0.5:
                    img = img.transpose((1, 2, 0))        #   h, w ,c
                    img = rotate(img)
                    img = img.transpose((2, 0, 1))   #  c, h, w
        img =np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        sample = {'image':img, 'label':label}
        return sample 

    def __len__(self):
            return len(self.dataframe)


class validation_strokelesion_Dataset(data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.dataframe = create_df(img_dir)
        self.dataframe.to_csv('df_validation.csv', sep='\t')

    def __getitem__(self, index):
        label = self.dataframe.iloc[index, 1]
        img = np.load(os.path.join(self.img_dir, self.dataframe.iloc[index, 0]))
        img = torch.from_numpy(img)
        sample = {'image': img, 'label': label}
        return sample

    def __len__(self):
        return len(self.dataframe)



