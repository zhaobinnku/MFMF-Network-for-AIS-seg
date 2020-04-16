import torch
import numpy as np
import SimpleITK  as sitk
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"]="1"
net = torch.load('./segmentation_process/model_best_segmentation_process.pkl')
net.eval()

def normalization(images):
    norm_images = np.zeros(images.shape)
    x = images[:,:,:]
    norm_images[:, :, :] = (x[:, :, :] - np.mean(x[:, :, :])) / (np.std(x[:, :, :]))  # 21张 整体标准化
    return norm_images


image_types = ['adc', 'dwi']
input_dir = './my_test_data/'
save_dir = './test_result/'
dirlist = os.listdir(input_dir)
pattern = re.compile(r'([0-9]+)')
reader = sitk.ImageFileReader()
patient = set()
for file_name in dirlist:
    search = pattern.search(file_name)
    if search is not None:
        patient.add(int(pattern.search(file_name).group(1)))
print(patient)
for patient_id in sorted(patient):

#*******************************************
    filename = patient_id
    individual_images = []
    nor_data = []
    for image_type in image_types:
        img_dir = os.path.join(input_dir, '%04d_%s.nii' % (patient_id, image_type))
        reader.SetFileName(img_dir)
        data = sitk.GetArrayFromImage(reader.Execute())
        data = normalization(data)
        individual_images.append(data)

    data = np.stack([image for image in individual_images],3)
    print(data.shape)
    data=data.transpose((0,3,1,2))
    predict_list =[]   # one slice input the net
    for i in range(data.shape[0]):
        tensor = torch.from_numpy(np.expand_dims(data[i,:,:,:], 0))
        img_variable = tensor.float().cuda()
        predict = net(img_variable)
        predict_np = predict.data.cpu().numpy()   # unet
        predict_np = predict_np.reshape(192, 192)
        predict_list.append(predict_np)
    predict_np = np.stack(predict_list, 0)
    sitk.WriteImage(sitk.GetImageFromArray(predict_np),os.path.join(save_dir,str(filename)+'_seg'+'.nii.gz'))
