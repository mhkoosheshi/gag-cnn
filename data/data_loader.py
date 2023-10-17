import numpy as np
from PIL import Image
import os
import math
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
import random
from tensorflow.keras.utils import Sequence
from data.path_lists import train_path_lists, test_path_lists, unison_shuffle
import albumentations as A
from utils.maps import rect2maps
from utils.utils import crop_object, crop_maps, ImageToFloatArray

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
              mode='rgb',
              aug_p=0.7,
              geo_p=0.5,
              color_p=0.5,
              noise_p=0.5,
              iso_p=0.5,
              stack=False,
              crop=False,
              maps=None
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
    self.crop = crop
    self.maps = maps
    self.mode = mode

    self.geo = A.Compose([
      A.HorizontalFlip(p=0.5)
    ], p=geo_p)
    
    self.noise = A.Compose([
      A.GaussNoise(p=0.5),
      A.MultiplicativeNoise(p=0.5),
    ], p=noise_p)

    self.color = A.Compose([
      A.RandomBrightness(p=0.5),
      A.FancyPCA(p=0.3),
      A.RandomShadow(p=0.2, shadow_roi=(0, 0.7, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4),
      A.RandomToneCurve(p=0.3),
      A.Solarize(threshold=50, p=0.5),
      A.PixelDropout(drop_value=0, dropout_prob=0.02, p=0.5),
      A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50, p=0.7)
    ], p=color_p) 

    self.transform = A.Compose([
    A.ISONoise(p=iso_p, color_shift=(0.01, 0.05), intensity=(0.2, 0.5)),
    self.noise,
    self.color,
    self.geo
    ], p=aug_p)

  def on_epoch_end(self):
    if self.shuffle:
      ind = np.random.permutation(len(self.RGBobj_paths)).astype(np.int64)
      self.RGBobj_paths, self.RGBiso_paths,  self.grasp_paths = np.array(self.RGBobj_paths), np.array(self.RGBiso_paths), np.array(self.grasp_paths)
      self.RGBobj_paths, self.RGBiso_paths, self.grasp_paths = self.RGBobj_paths[ind], self.RGBiso_paths[ind], self.grasp_paths[ind]
      self.RGBobj_paths, self.RGBiso_paths, self.grasp_paths = list(self.RGBobj_paths), list(self.RGBiso_paths), list(self.grasp_paths)


  def __len__(self):
    return math.ceil(len(self.RGBobj_paths) / self.batch_size)
  
  def bs(self):
    return self.batch_size


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
      # object bounding box here
      if self.crop:
        img, params = crop_object(img)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      # img = np.float32(img)
      img = img
      Q, W, Sin, Cos, Z = rect2maps(grasp_path)
      Q = np.asarray(Q, np.float64)
      W = np.asarray(W, np.float64)
      Sin = np.asarray(Sin, np.float64)
      Cos = np.asarray(Cos, np.float64)
      Z = np.asarray(Z, np.float64)
      
      if self.crop:
        Q = crop_maps(Q, params)
        W = crop_maps(W, params)
        Sin = crop_maps(Sin, params)
        Cos = crop_maps(Cos, params)

      Q = cv2.resize(Q, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
      W = cv2.resize(W, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
      Sin = cv2.resize(Sin, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
      Cos = cv2.resize(Cos, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
      Z = cv2.resize(Z, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img
        a = int(100*(random.random()))
        random.seed(a)
        transformed = self.transform(image=img, masks=[Q, W, Sin, Cos, Z])
        Q = transformed['masks'][0]
        W = transformed['masks'][1]
        Sin = transformed['masks'][2]
        Cos = transformed['masks'][3]
        Z = transformed['masks'][4]
        img = transformed['image']
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img

      # img = np.float32(img)
      rgbobj.append(img)
      Qmaps.append(Q)
      Wmaps.append(W)
      Sinmaps.append(Sin)
      Cosmaps.append(Cos)
      Zmaps.append(Z)


      # RGB2 data
      if self.mode=='rgb':
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
          transformed = self.transform(image=img)['image']
          img = transformed
          rnd = random.randint(1,2)
          rnd = rnd - 1
          img = (rnd)*(255 - img) + (1-rnd)*img
        
      elif self.mode == 'd':
        img = ImageToFloatArray(RGBiso_path)
        pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
        img = np.asarray(pimg)
        # img = np.float32(img)
        img = np.stack([img, img, img], axis=-1)

      # img = np.float32(img)
      rgbiso.append(img)
      # print(a)

    rgbobj = (np.array(rgbobj))/255
    if self.mode=='rgb':
      rgbiso = (np.array(rgbiso))/255
    elif self.mode=='d':
      rgbiso = (np.array(rgbiso))/22
    Qmaps = np.array(Qmaps)
    Wmaps = np.array(Wmaps)
    Sinmaps = np.array(Sinmaps)
    Cosmaps = np.array(Cosmaps)
    Zmaps = np.array(Zmaps)

    if self.maps is None:
      if self.stack:
        outputs = np.stack([Qmaps, Wmaps, Sinmaps, Cosmaps], axis=-1)
        return [rgbobj, rgbiso], [outputs]
      
      if not self.stack:
        return [rgbobj, rgbiso], [Qmaps, Wmaps, Sinmaps, Cosmaps]

    if self.maps is not None:
      if self.maps == 'Q':
        return [rgbobj, rgbiso], [Qmaps]

      if self.maps == 'W':
        return [rgbobj, rgbiso], [Wmaps]

      if self.maps == 'Sin':
        return [rgbobj, rgbiso], [Sinmaps]

      if self.maps == 'Cos':
        return [rgbobj, rgbiso], [Cosmaps]
      
      if self.maps == 'Angle':
        return [rgbobj, rgbiso], [Sinmaps, Cosmaps]

class DataGenerator2(Sequence):
  '''
  provide your model with batches of inputs and outputs with keras.utils.sequence

  two branches of RGB inputs for sided cameras
  '''
  def __init__(self,
              RGBobj_paths,
              RGBiso_paths,
              RGBiso2_paths,
              grasp_paths,
              batch_size=8,
              shape=(224,224),
              shuffle=True,
              mode='rgb',
              aug_p=0.7,
              geo_p=0.5,
              color_p=0.5,
              noise_p=0.5,
              iso_p=0.5,
              stack=False,
              crop=False,
              maps=None
              ):

    self.RGBobj_paths = RGBobj_paths
    self.RGBiso_paths = RGBiso_paths
    self.RGBiso2_paths = RGBiso2_paths
    self.grasp_paths = grasp_paths
    self.batch_size = batch_size
    self.shape = shape
    self.shuffle = shuffle
    self.aug_p = aug_p
    self.on_epoch_end()
    self.stack = stack
    self.crop = crop
    self.maps = maps
    self.mode = mode

    self.geo = A.Compose([
      A.HorizontalFlip(p=0.5)
    ], p=geo_p)
    
    self.noise = A.Compose([
      A.GaussNoise(p=0.5),
      A.MultiplicativeNoise(p=0.5),
    ], p=noise_p)

    self.color = A.Compose([
      A.RandomBrightness(p=0.5),
      A.FancyPCA(p=0.3),
      A.RandomShadow(p=0.2, shadow_roi=(0, 0.7, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4),
      A.RandomToneCurve(p=0.3),
      A.Solarize(threshold=50, p=0.5),
      A.PixelDropout(drop_value=0, dropout_prob=0.02, p=0.5),
      A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50, p=0.7)
    ], p=color_p) 

    self.transform = A.Compose([
    A.ISONoise(p=iso_p, color_shift=(0.01, 0.05), intensity=(0.2, 0.5)),
    self.noise,
    self.color,
    self.geo
    ], p=aug_p)

  def on_epoch_end(self):
    if self.shuffle:
      ind = np.random.permutation(len(self.RGBobj_paths)).astype(np.int64)
      self.RGBobj_paths, self.RGBiso_paths,  self.grasp_paths = np.array(self.RGBobj_paths), np.array(self.RGBiso_paths), np.array(self.grasp_paths)
      self.RGBobj_paths, self.RGBiso_paths, self.grasp_paths = self.RGBobj_paths[ind], self.RGBiso_paths[ind], self.grasp_paths[ind]
      self.RGBobj_paths, self.RGBiso_paths, self.grasp_paths = list(self.RGBobj_paths), list(self.RGBiso_paths), list(self.grasp_paths)


  def __len__(self):
    return math.ceil(len(self.RGBobj_paths) / self.batch_size)
  
  def bs(self):
    return self.batch_size


  def __getitem__(self, idx):

    batch_RGBobj = self.RGBobj_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_RGBiso = self.RGBiso_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_RGBiso2 = self.RGBiso2_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_grasp = self.grasp_paths[idx * self.batch_size : (idx+1) * self.batch_size]

    rgbobj = []
    rgbiso = []
    rgbiso2 = []
    Qmaps = []
    Wmaps = []
    Sinmaps =[]
    Cosmaps = []
    Zmaps = []

    for i, (RGBobj_path, RGBiso_path, RGBiso2_path, grasp_path) in enumerate(zip(batch_RGBobj, batch_RGBiso, batch_RGBiso2, batch_grasp)):
      
      # RGB1 data
      img = cv2.cvtColor(cv2.imread(RGBobj_path), cv2.COLOR_BGR2RGB)
      # object bounding box here
      if self.crop:
        img, params = crop_object(img)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      # img = np.float32(img)
      img = img
      Q, W, Sin, Cos, Z = rect2maps(grasp_path)
      Q = np.asarray(Q, np.float64)
      W = np.asarray(W, np.float64)
      Sin = np.asarray(Sin, np.float64)
      Cos = np.asarray(Cos, np.float64)
      Z = np.asarray(Z, np.float64)
      
      if self.crop:
        Q = crop_maps(Q, params)
        W = crop_maps(W, params)
        Sin = crop_maps(Sin, params)
        Cos = crop_maps(Cos, params)

      Q = cv2.resize(Q, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
      W = cv2.resize(W, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
      Sin = cv2.resize(Sin, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
      Cos = cv2.resize(Cos, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)
      Z = cv2.resize(Z, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_NEAREST)

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img
        a = int(100*(random.random()))
        random.seed(a)
        transformed = self.transform(image=img, masks=[Q, W, Sin, Cos, Z])
        Q = transformed['masks'][0]
        W = transformed['masks'][1]
        Sin = transformed['masks'][2]
        Cos = transformed['masks'][3]
        Z = transformed['masks'][4]
        img = transformed['image']
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img

      # img = np.float32(img)
      rgbobj.append(img)
      Qmaps.append(Q)
      Wmaps.append(W)
      Sinmaps.append(Sin)
      Cosmaps.append(Cos)
      Zmaps.append(Z)


      # RGB2 data
      if self.mode=='rgb':
        img = cv2.cvtColor(cv2.imread(RGBiso_path), cv2.COLOR_BGR2RGB)
        pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
        img = np.asarray(pimg)
        # img = np.float32(img)
        img = img

        img2 = cv2.cvtColor(cv2.imread(RGBiso2_path), cv2.COLOR_BGR2RGB)
        pimg2 = (Image.fromarray(img2)).resize((self.shape[0], self.shape[1]))
        img2 = np.asarray(pimg2)
        # img = np.float32(img)
        img2 = img2
        
        if self.aug_p !=0:
          rnd = random.randint(1,2)
          rnd = rnd - 1
          img = (rnd)*(255 - img) + (1-rnd)*img
          random.seed(a)
          transformed = self.transform(image=img)['image']
          img = transformed
          rnd = random.randint(1,2)
          rnd = rnd - 1
          img = (rnd)*(255 - img) + (1-rnd)*img
        
      elif self.mode == 'd':
        img = ImageToFloatArray(RGBiso_path)
        pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
        img = np.asarray(pimg)
        # img = np.float32(img)
        img = np.stack([img, img, img], axis=-1)

      # img = np.float32(img)
      rgbiso.append(img)
      rgbiso2.append(img2)
      # print(a)

    rgbobj = (np.array(rgbobj))/255
    if self.mode=='rgb':
      rgbiso = (np.array(rgbiso))/255
      rgbiso2 = (np.array(rgbiso))/255
    elif self.mode=='d':
      rgbiso = (np.array(rgbiso))/22
    Qmaps = np.array(Qmaps)
    Wmaps = np.array(Wmaps)
    Sinmaps = np.array(Sinmaps)
    Cosmaps = np.array(Cosmaps)
    Zmaps = np.array(Zmaps)

    if self.maps is None:
      if self.stack:
        outputs = np.stack([Qmaps, Wmaps, Sinmaps, Cosmaps], axis=-1)
        return [rgbobj, rgbiso, rgbiso2], [outputs]
      
      if not self.stack:
        return [rgbobj, rgbiso, rgbiso2], [Qmaps, Wmaps, Sinmaps, Cosmaps]

    if self.maps is not None:
      if self.maps == 'Q':
        return [rgbobj, rgbiso, rgbiso2], [Qmaps]

      if self.maps == 'W':
        return [rgbobj, rgbiso, rgbiso2], [Wmaps]

      if self.maps == 'Sin':
        return [rgbobj, rgbiso, rgbiso2], [Sinmaps]

      if self.maps == 'Cos':
        return [rgbobj, rgbiso, rgbiso2], [Cosmaps]
      
      if self.maps == 'Angle':
        return [rgbobj, rgbiso, rgbiso2], [Sinmaps, Cosmaps]

def get_loader(batch_size=8,
              mode='rgb',
              shape=(224,224),
              shuffle=True,
              factor=0.15,
              aug=False,
              aug_p=0,
              geo_p=0.5,
              color_p=0.5,
              noise_p=0.5,
              iso_p=0.5,
              val_aug_p=0,
              stack=False,
              crop=False,
              dataset_factor=1.0,
              maps=None,
              iso_num=1):
    # currently only for rgb and d
    if iso_num==1:
      RGBobj_paths, RGBiso_paths, grasp_paths = train_path_lists(mode=mode)
      n = len(RGBobj_paths)
      RGBobj_paths, RGBiso_paths, grasp_paths = np.array(RGBobj_paths), np.array(RGBiso_paths), np.array(grasp_paths)
      RGBobj_paths, RGBiso_paths, grasp_paths = unison_shuffle(a=RGBobj_paths, b=RGBiso_paths, c=grasp_paths)
      RGBobj_paths, RGBiso_paths, grasp_paths = list(RGBobj_paths), list(RGBiso_paths), list(grasp_paths)
      RGBobj_paths, RGBiso_paths, grasp_paths = RGBobj_paths[:int(n*dataset_factor)], RGBiso_paths[:int(n*dataset_factor)], grasp_paths[:int(n*dataset_factor)]
      RGBobj_train, RGBobj_val = RGBobj_paths[int(n*factor):], RGBobj_paths[:int(n*factor)]
      RGBiso_train, RGBiso_val = RGBiso_paths[int(n*factor):], RGBiso_paths[:int(n*factor)]
      grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]

      if aug:
          RGBobj_train, RGBiso_train, grasp_train = 2*RGBobj_train, 2*RGBiso_train, 2*grasp_train

      RGBobj_test, RGBiso_test, grasp_test = test_path_lists(mode=mode)

      train_gen = DataGenerator(RGBobj_train,
                                  RGBiso_train, 
                                  grasp_train,
                                  batch_size=batch_size,
                                  shape=shape,
                                  mode=mode,
                                  shuffle=shuffle,
                                  aug_p=aug_p,
                                  geo_p=geo_p,
                                  color_p=color_p,
                                  noise_p=noise_p,
                                  iso_p=iso_p,
                                  stack=stack,
                                  crop=crop,
                                  maps=maps
                                  )

      val_gen = DataGenerator(RGBobj_val,
                              RGBiso_val, 
                              grasp_val,
                              batch_size=batch_size,
                              shape=shape,
                              mode=mode,
                              shuffle=shuffle,
                              aug_p=val_aug_p,
                              geo_p=geo_p,
                              color_p=color_p,
                              noise_p=noise_p,
                              iso_p=iso_p,
                              stack=stack,
                              crop=crop,
                              maps=maps
                              )
      
      test_gen = DataGenerator(RGBobj_test,
                              RGBiso_test, 
                              grasp_test,
                              batch_size=batch_size,
                              shape=shape,
                              mode=mode,
                              shuffle=False,
                              aug_p=0,
                              geo_p=0,
                              color_p=0,
                              noise_p=0,
                              iso_p=0,
                              stack=stack,
                              crop=crop,
                              maps=maps
                              )

      return train_gen, val_gen, test_gen
    
    elif iso_num==2:
      RGBobj_paths, RGBiso_paths, RGBiso2_paths, grasp_paths = train_path_lists(mode=mode, iso_num=2)

      n = len(RGBobj_paths)
      RGBobj_paths, RGBiso_paths, RGBiso2_paths, grasp_paths = np.array(RGBobj_paths), np.array(RGBiso_paths), np.array(RGBiso2_paths), np.array(grasp_paths)
      RGBobj_paths, RGBiso_paths, RGBiso2_paths, grasp_paths = unison_shuffle(a=RGBobj_paths, b=RGBiso_paths, c=RGBiso2_paths, d=grasp_paths)
      RGBobj_paths, RGBiso_paths, RGBiso2_paths, grasp_paths = list(RGBobj_paths), list(RGBiso_paths), list(RGBiso2_paths), list(grasp_paths)
      RGBobj_paths, RGBiso_paths, RGBiso2_paths, grasp_paths = RGBobj_paths[:int(n*dataset_factor)], RGBiso_paths[:int(n*dataset_factor)], RGBiso2_paths[:int(n*dataset_factor)], grasp_paths[:int(n*dataset_factor)]
      RGBobj_train, RGBobj_val = RGBobj_paths[int(n*factor):], RGBobj_paths[:int(n*factor)]
      RGBiso_train, RGBiso_val = RGBiso_paths[int(n*factor):], RGBiso_paths[:int(n*factor)]
      RGBiso2_train, RGBiso2_val = RGBiso2_paths[int(n*factor):], RGBiso2_paths[:int(n*factor)]
      grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]

      if aug:
          RGBobj_train, RGBiso_train, grasp_train = 2*RGBobj_train, 2*RGBiso_train, 2*grasp_train

      RGBobj_test, RGBiso_test, RGBiso2_test, grasp_test = test_path_lists(mode=mode, iso_num=2)

      train_gen = DataGenerator2(RGBobj_train,
                                  RGBiso_train, 
                                  RGBiso2_train, 
                                  grasp_train,
                                  batch_size=batch_size,
                                  shape=shape,
                                  mode=mode,
                                  shuffle=shuffle,
                                  aug_p=aug_p,
                                  geo_p=geo_p,
                                  color_p=color_p,
                                  noise_p=noise_p,
                                  iso_p=iso_p,
                                  stack=stack,
                                  crop=crop,
                                  maps=maps
                                  )

      val_gen = DataGenerator2(RGBobj_val,
                              RGBiso_val, 
                              RGBiso2_val, 
                              grasp_val,
                              batch_size=batch_size,
                              shape=shape,
                              mode=mode,
                              shuffle=shuffle,
                              aug_p=val_aug_p,
                              geo_p=geo_p,
                              color_p=color_p,
                              noise_p=noise_p,
                              iso_p=iso_p,
                              stack=stack,
                              crop=crop,
                              maps=maps
                              )
      
      test_gen = DataGenerator2(RGBobj_test,
                              RGBiso_test, 
                              RGBiso2_test, 
                              grasp_test,
                              batch_size=batch_size,
                              shape=shape,
                              mode=mode,
                              shuffle=False,
                              aug_p=0,
                              geo_p=0,
                              color_p=0,
                              noise_p=0,
                              iso_p=0,
                              stack=stack,
                              crop=crop,
                              maps=maps
                              )

      return train_gen, val_gen, test_gen
