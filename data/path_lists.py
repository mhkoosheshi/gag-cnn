import numpy as np
import glob


def train_path_lists(mode='rgb'):
  '''
  mode : 'rgb' or 'rgbd' or 'd'
  '''
  grasps = []
  rgb_obj = []
  d_obj = []
  rgb_iso = []
  d_iso = []
  
  for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/grasp/grasp*.txt")):
      grasps.append(grasp_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/obj/rgb*.png"):
      rgb_obj.append(im_path)
  for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs10/rgb*.png"):
      rgb_iso.append(im_path)
  
  
  if 'd' in mode:
    for d_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/vs10/d*.png"):
      d_iso.append(d_path)
    for d_path in glob.glob(f"/content/drive/MyDrive/gag-cnn/obj/d*.png"):
      d_obj.append(d_path)
  
  grasps = sorted(grasps)
  rgb_obj = sorted(rgb_obj)
  d_obj = sorted(d_obj)
  rgb_iso = sorted(rgb_iso)
  d_iso = sorted(d_iso)

  if mode=='rgb':
    return rgb_obj, rgb_iso, grasps

def test_path_lists():
  # fix pers and iso labeling
    list_grasp = []
    list_RGB1 = []
    list_D1 = []
    list_RGB2 = []
    list_D2 = []
    list_RGB3 = []
    list_D3 = []

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs1/rgb*.png"):
        list_RGB1.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/test/grasp/grasp*.txt")):
        list_grasp.append(grasp_path)

    for d_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs1/d*.png"):
        list_D1.append(d_path)

    list_RGB1 = sorted(list_RGB1)
    list_grasp = sorted(list_grasp)
    list_D1 = sorted(list_D1)

    return list_RGB1, list_D1, list_grasp

def unison_shuffle(a, b, c, d=None):
  
  if d is None:
    np.random.seed(42)
    inx=np.random.permutation(a.shape[0])
    return a[inx],b[inx],c[inx]
  elif d is not None:
    np.random.seed(42)
    inx=np.random.permutation(a.shape[0])
    return a[inx], b[inx], c[inx], d[inx]
