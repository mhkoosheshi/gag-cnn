import numpy as np
from PIL import Image
import os
import math
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
import random
from tensorflow.keras.utils import Sequence
from path_lists import train_path_lists, test_path_lists, unison_shuffle
import albumentations as A
from utils import rect2maps

class DataGenerator(Sequence):
  '''
  provide your model with batches of inputs and outputs with keras.utils.sequence

  two branches of RGB inputs for sided cameras
  '''
  def __init__(self,
              RGBobj_paths,
              RGBiso_paths,
              grasp_paths,
              batch_size=8,
              shape=(224,224),
              shuffle=True,
              aug_p=0.7,
              geo_p=0.5,
              color_p=0.5,
              noise_p=0.5,
              others_p=0.5,
              iso_p=0.5,
              stack=False,
              ):

    self.RGBobj_paths = RGBobj_paths
    self.RGBiso_paths = RGBiso_paths
    self.grasp_paths = grasp_paths
    self.batch_size = batch_size
    self.shape = shape
    self.shuffle = shuffle
    self.aug_p = aug_p
    self.on_epoch_end()
    self.stack = stack

    self.geo = A.Compose([


    ], p=geo_p)
    
    self.noise = A.Compose([
      A.GaussNoise(p=0.5),
      A.MultiplicativeNoise(p=0.5),
    ], p=noise_p)

    self.others = A.Compose([
      A.RandomBrightness(p=0.5),
      A.FancyPCA(p=0.3),
      A.RandomShadow(p=0.2, shadow_roi=(0, 0.7, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4),
      A.RandomToneCurve(p=0.3),
      A.Solarize(threshold=50, p=0.5),
      A.PixelDropout(drop_value=0, dropout_prob=0.02, p=0.5),
      A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50, p=0.7)
    ], p=others_p) 

    self.color = A.Compose([
    A.ISONoise(p=iso_p, color_shift=(0.01, 0.05), intensity=(0.2, 0.5)),
    self.others,
    self.noise
    ], p=color_p)

    self.transform = A.Compose([
    ], p=aug_p)

  def on_epoch_end(self):
    if self.shuffle:
      ind = np.random.permutation(len(self.RGBobj_paths)).astype(np.int64)
      self.RGBobj_paths, self.RGBiso_paths,  self.grasp_paths = np.array(self.RGBobj_paths), np.array(self.RGBiso_paths), np.array(self.grasp_paths)
      self.RGBobj_paths, self.RGBiso_paths, self.grasp_paths = self.RGBobj_paths[ind], self.RGBiso_paths[ind], self.grasp_paths[ind]
      self.RGBobj_paths, self.RGBiso_paths, self.grasp_paths = list(self.RGBobj_paths), list(self.RGBiso_paths), list(self.grasp_paths)


  def __len__(self):
    return math.ceil(len(self.RGBobj_paths) / self.batch_size)


  def __getitem__(self, idx):

    batch_RGBobj = self.RGBobj_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_RGBiso = self.RGBiso_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_grasp = self.grasp_paths[idx * self.batch_size : (idx+1) * self.batch_size]

    rgbobj = []
    rgbiso = []
    Qmaps = []
    Wmaps = []
    Sinmaps =[]
    Cosmaps = []
    Zmaps = []

    for i, (RGBobj_path, RGBiso_path, grasp_path) in enumerate(zip(batch_RGBobj, batch_RGBiso, batch_grasp)):
      
      # RGB1 data
      img = cv2.cvtColor(cv2.imread(RGBobj_path), cv2.COLOR_BGR2RGB)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      # img = np.float32(img)
      img = img

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img
        a = int(100*(random.random()))
        random.seed(a)
        transformed = self.color_transform(image=img)['image']
        img = transformed

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img

      # img = np.float32(img)
      rgbobj.append(img)


      # RGB2 data
      img = cv2.cvtColor(cv2.imread(RGBiso_path), cv2.COLOR_BGR2RGB)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      # img = np.float32(img)
      img = img
      
      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img
        random.seed(a)
        transformed = self.color_transform(image=img)['image']
        img = transformed

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img

      # img = np.float32(img)
      rgbiso.append(img)

      # translate grasp data into grasping maps
      Q, W, Sin, Cos, Z = rect2maps(grasp_path)
      Qmaps.append(Q)
      Wmaps.append(W)
      Sinmaps.append(Sin)
      Cosmaps.append(Cos)
      Zmaps.append(Z)
      # print(a)

    rgbobj = (np.array(rgbobj))/255
    rgbiso = (np.array(rgbiso))/255
    Qmaps = np.array(Qmaps)
    Wmaps = np.array(Wmaps)
    Sinmaps = np.array(Sinmaps)
    Cosmaps = np.array(Cosmaps)
    Zmaps = np.array(Zmaps)

    if self.stack:
      outputs = np.stack([Qmaps, Wmaps, Sinmaps, Cosmaps], axis=-1)
      return [rgbobj, rgbiso], [outputs]
    
    if not self.stack:
      return [rgbobj, rgbiso], [Qmaps, Wmaps, Sinmaps, Cosmaps]

def get_loader(batch_size=8,
              mode='rgb',
              shape=(224,224),
              shuffle=True,
              factor=0.15,
              aug=False,
              aug_p=0,
              stack=False):
    # currently only for rgb
    RGBobj_paths, RGBiso_paths, grasp_paths = train_path_lists(mode=mode)
    n = len(RGBobj_paths)
    RGBobj_paths, RGBiso_paths, grasp_paths = np.array(RGBobj_paths), np.array(RGBiso_paths), np.array(grasp_paths)
    RGBobj_paths, RGBiso_paths, grasp_paths = unison_shuffle(a=RGBobj_paths, b=RGBiso_paths, c=grasp_paths)
    RGBobj_paths, RGBiso_paths, grasp_paths = list(RGBobj_paths), list(RGBiso_paths), list(grasp_paths)
    RGBobj_train, RGBobj_val = RGBobj_paths[int(n*factor):], RGBobj_paths[:int(n*factor)]
    RGBiso_train, RGBiso_val = RGBiso_paths[int(n*factor):], RGBiso_paths[:int(n*factor)]
    grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]

    if aug:
        RGBobj_train, RGBiso_train, grasp_train = 2*RGBobj_train, 2*RGBiso_train, 2*grasp_train

    # RGB1_test, RGB2_test, RGB3_test, grasp_test = test_path_lists(mode=mode)

    train_gen = DataGenerator(RGBobj_train,
                                RGBiso_train, 
                                grasp_train,
                                batch_size=batch_size,
                                shape=shape,
                                shuffle=shuffle,
                                aug_p=aug_p,
                                stack=stack
                                )
    val_gen = DataGenerator(RGBobj_val,
                            RGBiso_val, 
                            grasp_val,
                            batch_size=batch_size,
                            shape=shape,
                            shuffle=shuffle,
                            aug_p=0,
                            stack=stack
                            )

    # test_gen = DataGenerator(RGB1_test,
    #                             RGB2_test, 
    #                             RGB3_test,
    #                             grasp_test,
    #                             batch_size=batch_size,
    #                             shape=shape,
    #                             shuffle=False,
    #                             aug_p=0
    #                             )

    return train_gen, val_gen