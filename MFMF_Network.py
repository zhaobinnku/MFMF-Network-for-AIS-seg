import torch
import numpy as np
import torch.utils.data as data
import os
import pandas as pd
import random
import skimage.transform as transform
import torch.nn as nn

vgg16_pretrain = torch.load('./classification_process/model_best_classification_process.pkl')
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def rotate(img, label):
    rotate_degree = random.randint(1, 360)
    return  transform.rotate(img, rotate_degree, mode='reflect'),  transform.rotate(label, rotate_degree, mode='reflect')

def HorizontallyFlip(img, label):
    img = img[:, :, :: -1]
    label = label[:, :, :: -1]
    return img.copy(), label.copy()

def VerticallyFlip(img, label):
    img = img[:, ::-1, :]
    label = label[:, ::-1, :]
    return img.copy(), label.copy()

def create_df(img_dir):
    """Create pd dataframe for dataloader
    Args:
        img_dir (str): directory containing the images
    Returns:
        df: pd dataframe
    """
    img_lis = os.listdir(img_dir)
    case_lis = []
    label_lis = []

    for filename in img_lis:
        if filename.endswith('.npy') and 'label' not in filename:
            fn1 = filename.split('_')[1]
            fn2 = filename.split('_')[1][1]
            if str(fn2) == '.':
                fn3 = filename.split('_')[1][0]
            else:
                fn3 = filename.split('_')[1][0:2]
            mask_name = filename.replace(str(fn1), str(fn3) + 'label.npy')
            case_lis.append(filename)
            label_lis.append(mask_name)
    df_r = pd.DataFrame({
        'casename': case_lis, 'label': label_lis
    })
    return df_r


class train_dataset_segmentation(data.Dataset):
    def __init__(self,img_dir, transforms):
        self.img_dir = img_dir
        self.dataframe = create_df(img_dir)
        self.transforms = transforms
        self.dataframe.to_csv('df_train_segmentation.csv', sep='\t')

    def __getitem__(self, index):
        label  = np.load(os.path.join(self.img_dir, self.dataframe.iloc[index,1]))
        label = np.expand_dims(label, axis=0)  #  add channel   (1, 192, 192)
        image  = np.load(os.path.join(self.img_dir, self.dataframe.iloc[index,0]))
        if self.transforms:
            if np.random.rand() < 0.5:
                if np.random.rand() > 0.5:
                    image, label = HorizontallyFlip(image, label)

            if np.random.rand() < 0.5:
                if np.random.rand() >0.5:
                    image, label = VerticallyFlip(image, label)

            if np.random.rand() < 0.5:
                if np.random.rand() > 0.5:
                    image = image.transpose((1,2,0))
                    label = label.transpose((1,2,0))
                    image, label = rotate(image, label)
                    image = image.transpose((2,0,1))
                    label = label.transpose((2,0,1))
        sample = {'image': image.astype(np.float), 'label': label.astype(np.float)}
        return sample

    def __len__(self):
        return len(self.dataframe)


class validation_dataset_segmentation(data.Dataset):
    def __init__(self,img_dir, transforms):
        self.img_dir = img_dir
        self.dataframe = create_df(img_dir)
        self.transforms = transforms
        self.dataframe.to_csv('df_validation_segmentation.csv', sep='\t')

    def __getitem__(self, index):
        label  = np.load(os.path.join(self.img_dir, self.dataframe.iloc[index,1]))
        label = np.expand_dims(label, axis=0)  #  add channel   (1, 192, 192)
        image  = np.load(os.path.join(self.img_dir, self.dataframe.iloc[index,0]))
        if self.transforms:
            if np.random.rand() < 0.5:
                if np.random.rand() > 0.5:
                    image, label = HorizontallyFlip(image, label)

            if np.random.rand() < 0.5:
                if np.random.rand() >0.5:
                    image, label = VerticallyFlip(image, label)

            if np.random.rand() < 0.5:
                if np.random.rand() > 0.5:
                    image = image.transpose((1,2,0))
                    label = label.transpose((1,2,0))
                    image, label = rotate(image, label)
                    image = image.transpose((2,0,1))
                    label = label.transpose((2,0,1))
        sample = {'image': image.astype(np.float), 'label': label.astype(np.float)}
        return sample

    def __len__(self):
        return len(self.dataframe)


class VGG16(nn.Module):
    def __init__(self, init_weights=False):
        super(VGG16, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.features3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.features4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.features5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.features6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.features7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.features8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features11 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.features13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        for p in self.parameters():   # no grad flow
            p.requires_grad = False
        # ********   128 channels  from feature layer 4***********
        self.upsample4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.gap4 = nn.AdaptiveAvgPool2d(1)
        self.in_fc4 = nn.Linear(128, 8)
        self.out_fc4 = nn.Linear(8, 128)
         #********   256 channels  from feature layer 7***********
        self.gap7 = nn.AdaptiveAvgPool2d(1)
        self.in_fc7 = nn.Linear(256, 16)
        self.out_fc7 = nn.Linear(16, 256)
        self.relu7 = nn.ReLU(inplace=True)
        self.upsample7 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7= nn.ReLU(inplace=True)
        # ********   512 channels  from feature layer 10***********
        self.gap10 = nn.AdaptiveAvgPool2d(1)
        self.in_fc10 = nn.Linear(512, 32)
        self.out_fc10 = nn.Linear(32, 512)
        self.upsample10 = nn.ConvTranspose2d(512, 512, kernel_size=8, stride=8)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(inplace=True)
        self.final = nn.Conv2d(896, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x6 = self.features6(x5)
        x7 = self.features7(x6)
        x8 = self.features8(x7)
        x9 = self.features9(x8)
        x10 = self.features10(x9)
        # ********   128 channels  from feature layer 4***********
        gap_x4 = self.gap4(x4)
        gap_x4 = gap_x4.view(gap_x4.size(0), -1)
        in_x4= self.relu4(self.in_fc4(gap_x4))
        out_4 = self.out_fc4(in_x4)
        sig_4 = self.sigmoid(out_4)
        sig_4 = sig_4.view(sig_4.size(0), sig_4.size(1), 1, 1)
        x4 = sig_4 * x4
        x4 = self.upsample4(x4)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)
        # ********   256 channels  from feature layer 7***********
        gap_x7 = self.gap7(x7)
        gap_x7 = gap_x7.view(gap_x7.size(0), -1)
        in_x7 = self.relu7(self.in_fc2(gap_x7))
        out_x7 = self.out_fc7(in_x7)
        sig_x7 = self.sigmoid(out_x7)
        sig_x7 = sig_x7.view(sig_x7.size(0), sig_x7.size(1), 1, 1)
        x7 = sig_x7 * x7
        x7 = self.upsample7(x7)
        x7 = self.bn7(x7)
        x7 = self.relu7(x7)
        # ********   512 channels  from feature layer 10***********
        gap_x10 = self.gap10(x10)
        gap_x10 = gap_x10.view(gap_x10.size(0), -1)
        in_x10 = self.relu10(self.in_fc1(gap_x10))
        out_10 = self.out_fc10(in_x10)
        sig_10 = self.sigmoid(out_10)
        sig_10 = sig_10.view(sig_10.size(0), sig_10.size(1), 1 , 1)
        x10 = sig_10 * x10
        x10 = self.upsample10(x10)
        x10 = self.bn10(x10)
        x10 = self.relu10(x10)
        # *****************************  more features fusion*************
        x = torch.cat((x4,x10, x7), 1)
        x = self.final(x)
        x = self.sigmoid(x)
        return x

    # initialize every parameters
    def _initialize_weights(self):
        print('initialing')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.ConvTranspose2d):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


class segmentaion_process(nn.Module):
    def __init__(self):
        super(segmentaion_process, self).__init__()
        vgg16 = vgg16_pretrain
        vgg_16 = VGG16()
        pretrained_dict = vgg16.state_dict()
        model_dict = vgg_16.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        vgg_16.load_state_dict(model_dict)
        self.vgg16_segmentation = vgg_16

    def forward(self, inputs):
        out = self.vgg16_segmentation(inputs)
        return out
