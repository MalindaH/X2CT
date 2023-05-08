# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from lib.dataset.baseDataSet import Base_DataSet
from lib.dataset.utils import *
import h5py
import numpy as np

# from deepdrr import Volume
import SimpleITK as sitk
import PIL

class AlignDataSet(Base_DataSet):
  '''
  DataSet For unaligned data
  '''
  def __init__(self, opt):
    super(AlignDataSet, self).__init__()
    self.opt = opt
    self.ext = '.h5'
    self.dataset_paths = get_dataset_from_txt_file(self.opt.datasetfile)
    self.dataset_paths = sorted(self.dataset_paths)
    self.dataset_size = len(self.dataset_paths)
    self.dir_root = self.get_data_path
    self.data_augmentation = self.opt.data_augmentation(opt)

  @property
  def name(self):
    return 'AlignDataSet'

  @property
  def get_data_path(self):
    path = os.path.join(self.opt.dataroot)
    return path

  @property
  def num_samples(self):
    return self.dataset_size

  def get_image_path(self, root, index_name):
    img_path = os.path.join(root, index_name, 'ct_xray_data'+self.ext)
    assert os.path.exists(img_path), 'Path do not exist: {}'.format(img_path)
    return img_path
    
  def load_file(self, file_path):
    hdf5 = h5py.File(file_path, 'r')
    ct_data = np.asarray(hdf5['ct'])
    x_ray1 = np.asarray(hdf5['xray1'])
    x_ray2 = np.asarray(hdf5['xray2'])
    x_ray1 = np.expand_dims(x_ray1, 0)
    x_ray2 = np.expand_dims(x_ray2, 0)
    hdf5.close()
    return ct_data, x_ray1, x_ray2
  
  def get_ct_path(self, root, filename): # filename: imagesTr/lung_022.nii.gz
    ct_path = root+filename
    # xray_path_front = root+filename[:8]+"_xray"+filename[8:-7]+"_front.png"
    # xray_path_side = root+filename[:8]+"_xray"+filename[8:-7]+"_side.png"
    xray_path_front = root+filename[:8]+"_xray1"+filename[8:-7]+"_front.png"
    xray_path_side = root+filename[:8]+"_xray1"+filename[8:-7]+"_side.png"
    # print(ct_path, xray_path_front, xray_path_side)
    assert os.path.exists(ct_path), 'Path do not exist: {}'.format(ct_path)
    assert os.path.exists(xray_path_front), 'Path do not exist: {}'.format(xray_path_front)
    assert os.path.exists(xray_path_side), 'Path do not exist: {}'.format(xray_path_side)
    return ct_path, xray_path_front, xray_path_side
  
  def load_images(self, ct_path, xray_path_front, xray_path_side):
    # ct = Volume.from_nifti(ct_path)
    # ct.orient_patient(head_first=True, supine=True)
    # print(ct)

    volume = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)) # old: (226, 512, 512) (dim 0 is different for each sample)
    volume = volume+1024
    # print(volume.shape) # (256, 256, 256)
    
    xray_front = np.asarray(PIL.Image.open(xray_path_front, mode='r')) # old: (480, 640, 4)
    xray_side = np.asarray(PIL.Image.open(xray_path_side, mode='r')) # (480, 640)
    xray_front = np.expand_dims(xray_front, 0) # (1, 480, 640)
    xray_side = np.expand_dims(xray_side, 0) # (1, 480, 640)
    # print(xray_front.shape, xray_side.shape)
    
    return volume, xray_front, xray_side


  '''
  generate batch
  '''
  def pull_item(self, item):
    if self.dir_root.endswith("LIDC-HDF5-256"):
      file_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
      ct_data, x_ray1, x_ray2 = self.load_file(file_path)
      
      # Data Augmentation
      ct, xray1, xray2 = self.data_augmentation([ct_data, x_ray1, x_ray2])
      
      return ct, xray1, xray2, file_path
    else:
      ct_path, xray_path_front, xray_path_side = self.get_ct_path(self.dir_root, self.dataset_paths[item])
      ct_data, x_ray1, x_ray2 = self.load_images(ct_path, xray_path_front, xray_path_side)

      # Data Augmentation
      ct, xray1, xray2 = self.data_augmentation([ct_data, x_ray1, x_ray2])

    # return ct, xray1, xray2, file_path
      return ct, xray1, xray2, ct_path






