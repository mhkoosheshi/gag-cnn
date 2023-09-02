import cv2
import numpy as np
from .maps import rect2maps, camera_calibration

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

def nearest_rect(x, y, true_grasp_path, shape=(512,512)):
    
    Q, _, _, _, _ = rect2maps(true_grasp_path, shape=shape)
    with open(true_grasp_path,"r") as f:
        s = f.read()
    grasp = [float(s.split(",")[i]) for i in range(0,len(s.split(",")))]
    grasp = camera_calibration(grasp, shape=shape)
    [x_c, y_c, _, t, width] = grasp

    X = shape[0]
    Y = shape[1]

    m = np.tan(t*np.pi/180)
    m_ = np.tan((t*np.pi/180)+(np.pi)/2)

    x_t = (y_c - y - m*x_c + m_*x)/(m_ - m)
    y_t = y_c + (x_t - x_c)*m_

    return [x_t, y_t, t, width]