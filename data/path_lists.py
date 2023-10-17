import numpy as np
import glob


def train_path_lists(mode='rgb', iso_num=1):
  '''
  mode : 'rgb' or 'rgbd' or 'd'
  '''
  grasps = []
  rgb_obj = []
  d_obj = []
  rgb_iso = []
  d_iso = []
  rgb_iso2 = []
  rgb_iso3 = []
  
  for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/grasp/grasp*.txt")):
      grasps.append(grasp_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/obj/rgb*.png"):
      rgb_obj.append(im_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs10/rgb*.png"):
      rgb_iso.append(im_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs11/rgb*.png"):
      rgb_iso2.append(im_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs12/rgb*.png"):
      rgb_iso3.append(im_path)
  
  
  if 'd' in mode:
    for d_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs10/d*.png"):
      d_iso.append(d_path)
    for d_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/obj/d*.png"):
      d_obj.append(d_path)
  
  grasps = sorted(grasps)
  rgb_obj = sorted(rgb_obj)
  d_obj = sorted(d_obj)
  rgb_iso = sorted(rgb_iso)
  rgb_iso2 = sorted(rgb_iso2)
  rgb_iso3 = sorted(rgb_iso3)
  d_iso = sorted(d_iso)

  if iso_num==1:
    if mode=='rgb':
      return rgb_obj, rgb_iso, grasps

    elif mode=='rgbd':
      return rgb_obj, rgb_iso, d_obj, d_iso, grasps

    elif mode=='d':
      return rgb_obj, d_iso, grasps
  
  elif iso_num==2:
      return rgb_obj, rgb_iso2, rgb_iso3, grasps

  elif iso_num==3:
      return rgb_obj, rgb_iso, rgb_iso2, rgb_iso3, grasps

def test_path_lists(mode='rgb', iso_num=1):

  grasps = []
  rgb_obj = []
  d_obj = []
  rgb_iso = []
  d_iso = []
  rgb_iso2 = []
  rgb_iso3 = []
  
  for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/gag-cnn/test/grasp/grasp*.txt")):
      grasps.append(grasp_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/test/obj/rgb*.png"):
      rgb_obj.append(im_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/test/vs10/rgb*.png"):
      rgb_iso.append(im_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/test/vs11/rgb*.png"):
      rgb_iso2.append(im_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/test/vs12/rgb*.png"):
      rgb_iso3.append(im_path)
  
  if 'd' in mode:
    for d_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/test/vs10/d*.png"):
      d_iso.append(d_path)
    for d_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/test/obj/d*.png"):
      d_obj.append(d_path)
  
  grasps = sorted(grasps)
  rgb_obj = sorted(rgb_obj)
  d_obj = sorted(d_obj)
  rgb_iso = sorted(rgb_iso)
  rgb_iso2 = sorted(rgb_iso2)
  rgb_iso3 = sorted(rgb_iso3)
  d_iso = sorted(d_iso)
  if iso_num==1:
    if mode=='rgb':
      return rgb_obj, rgb_iso, grasps
    
    elif mode=='rgbd':
      return rgb_obj, rgb_iso, d_iso, grasps

    elif mode=='d':
      return rgb_obj, d_iso, grasps

  elif iso_num==2:
    return rgb_obj, rgb_iso, rgb_iso2, grasps

def unison_shuffle(a, b, c, d=None):
  
  if d is None:
    np.random.seed(42)
    inx=np.random.permutation(a.shape[0])
    return a[inx],b[inx],c[inx]
  elif d is not None:
    np.random.seed(42)
    inx=np.random.permutation(a.shape[0])
    return a[inx], b[inx], c[inx], d[inx]
