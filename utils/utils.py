import cv2
import numpy as np
from .maps import rect2maps, camera_calibration
from utils.maps import rect2maps, Rotate_center, center2points

def crop_object(img):
    x = img.copy()
    x[:,:,0] = x[:,:,0] - 94
    x[:,:,1] = x[:,:,1] - 255
    x[:,:,2] = x[:,:,2] - 119

    x = x/255
    x = np.ceil(x)

    x = np.array(x, np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    x = cv2.dilate(x, kernel, iterations=8)
    cnts, _=cv2.findContours(x[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x0,y0,w0,h0 = cv2.boundingRect(cnts[0])
    if w0>256 or h0>256:
        w0 = max(w0,h0)
        h0 = max(w0,h0)
    else:
        w0 = 256
        h0 = 256 

    if x0+w0>x.shape[0]:
        x0 = x0 - (x0+w0-x.shape[0])
    if y0+h0>x.shape[1]:
        y0 = y0 - (y0+h0-x.shape[1])
    crop_params = [x0, y0, w0, h0]
    img1 = img[y0:y0+h0,x0:x0+w0]

    return img1, crop_params

def crop_maps(map, crop_params):
    [x0, y0, w0, h0] = crop_params
    map1 = map[y0:y0+h0,x0:x0+w0]
    return map1

# def nearest_rect(x, y, true_grasp_path, shape=(512,512)):
    
#     Q, _, _, _, _ = rect2maps(true_grasp_path, shape=shape)
#     with open(true_grasp_path,"r") as f:
#         s = f.read()
#     grasp = [float(s.split(",")[i]) for i in range(0,len(s.split(",")))]
#     grasp = camera_calibration(grasp, shape=shape)
#     [x_c, y_c, _, t, width] = grasp

#     X = shape[0]
#     Y = shape[1]

#     m = np.tan(t*np.pi/180)
#     m_ = np.tan((t*np.pi/180)+(np.pi)/2)

#     x_t = (y_c - y - m*x_c + m_*x)/(m_ - m)
#     y_t = y_c + (x_t - x_c)*m_

#     return [x_t, y_t, t, width]

def nearest_rect(x, y, grasp_path, num=5, shape=(512,512)):

  Q, _, _, _, _ = rect2maps(grasp_path, shape)

  with open(grasp_path,"r") as f:
        s = f.read()
  grasp = [float(s.split(",")[i]) for i in range(0,len(s.split(",")))]

  [x_c, y_c, z_c, t, w] = grasp
  y_c = ((0.5*y_c)/512 -0.0442 - 0.125 + 0.0079 + 0.05)/0.35*shape[0]
  x_c = ((0.5*x_c)/512 - 0.125 + 0.05)/0.35*shape[1]
  l=15
  l=l*(0.5/0.35)
  xp_c, yp_c = Rotate_center(x_c, y_c, t, w)

  grasps = [[xp_c, yp_c]]
  c=(l/2)/((num-1)/2)
  # c=l/2

  for i in range(0, int((num-1)/2)):
    grasps.append([xp_c + c*(i+1), yp_c])
    grasps.append([xp_c - c*(i+1), yp_c])

  grasps = np.array(grasps)
  euc = np.zeros((grasps.shape[0], 1))
  euc[:, 0] = np.sqrt((grasps[0, :]-x)*(grasps[0, :]-x) + (grasps[1, :]-y)*(grasps[1, :]-y))
  row = (np.unravel_index(np.argmin(euc, axis=None), euc.shape))[0]

  return grasps[row,0], grasps[row,1], t, w

def ImageToFloatArray(path, scale_factor=None):
    DEFAULT_RGB_SCALE_FACTOR = 256000.0
    DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                                 np.uint16: 1000.0,
                                 np.int32: DEFAULT_RGB_SCALE_FACTOR}
    image = cv2.imread(r"{}".format(path))
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array