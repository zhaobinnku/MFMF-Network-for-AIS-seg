import torch
import numpy as np
import SimpleITK  as sitk
import os
import re


os.environ["CUDA_VISIBLE_DEVICES"]="1"

cam_filedir = './CAM/'
seg_file_dir = './Seg/'
save_dir = './Final_result/'

dirlist = os.listdir(seg_file_dir)
pattern = re.compile(r'([0-9]+)')
reader = sitk.ImageFileReader()
patient = set()
for file_name in dirlist:
    search = pattern.search(file_name)
    if search is not None:
        patient.add(int(pattern.search(file_name).group(1)))
print(patient)

for patient_id in sorted(patient):
    print(patient_id)

#*******************************************
    # result = 0

    cam_dir = os.path.join(cam_filedir, '%04d_cam.nii' % (patient_id))
    reader.SetFileName(cam_dir)
    cam = sitk.GetArrayFromImage(reader.Execute())
    loc = cam < 0.5
    cam[loc] = 0
    loc = cam >=0.5
    cam[loc] = 1

    seg_dir = os.path.join(seg_file_dir, '%04d_seg.nii.gz' % (patient_id))
    reader.SetFileName(seg_dir)
    seg = sitk.GetArrayFromImage(reader.Execute())
    result = cam*seg
    print(result.shape)
    sitk.WriteImage(sitk.GetImageFromArray(result), os.path.join(save_dir, str(patient_id)+'_result.nii.gz'))





