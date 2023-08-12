import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from shapely.geometry import Polygon
from tensorflow.python.ops.control_flow_ops import tuple_v2

def draw_grasp(image_path: str, grasp_path:str, shape=(512,512)):
    x = cv2.imread(image_path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    image = x
    
    with open(grasp_path,"r") as f:
        s = f.read()
    grasp = [float(s.split(",")[i]) for i in range(0,len(s.split(",")))]

    if not ('obj2' in image_path):
        grasp = camera_calibration(grasp)
        [x_c, y_c, z_c, t, width] = grasp
    else:
        [x_c, y_c, z_c, t, width] = grasp
        y_c = ((0.5*y_c)/512 -0.0442 - 0.125 + 0.0079 + 0.05)/0.35*512
        x_c = ((0.5*x_c)/512 - 0.125 + 0.05)/0.35*512


    t = np.deg2rad(90-t)
    l = 15
    w = width/2
    w=(w*(0.5/0.35)*105/100)/4
    l=l*(0.5/0.35)*1.2


    R1 = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    R2 = np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    xp_c = (np.matmul(R1, np.array([[x_c],[y_c]])))[0][0]
    yp_c = (np.matmul(R1, np.array([[x_c],[y_c]])))[1][0]

    xp_1 = xp_c + l
    yp_1 = yp_c - w
    xp_2 = xp_1
    yp_2 = yp_c + w
    xp_3 = xp_c - l
    yp_3 = yp_2
    xp_4 = xp_3
    yp_4 = yp_1

    x1 = int((np.matmul(R2, np.array([[xp_1],[yp_1]])))[0][0])
    y1 = int((np.matmul(R2, np.array([[xp_1],[yp_1]])))[1][0])
    x2 = int((np.matmul(R2, np.array([[xp_2],[yp_2]])))[0][0])
    y2 = int((np.matmul(R2, np.array([[xp_2],[yp_2]])))[1][0])
    x3 = int((np.matmul(R2, np.array([[xp_3],[yp_3]])))[0][0])
    y3 = int((np.matmul(R2, np.array([[xp_3],[yp_3]])))[1][0])
    x4 = int((np.matmul(R2, np.array([[xp_4],[yp_4]])))[0][0])
    y4 = int((np.matmul(R2, np.array([[xp_4],[yp_4]])))[1][0])

    point1 = x1,y1
    point2 = x2,y2
    point3 = x3,y3
    point4 = x4,y4

    color1 = (0,0,200)
    color2 = (200,0,0)
    thickness = 4

    image = cv2.line(image, point1, point2, color1, thickness)
    image = cv2.line(image, point3, point4, color1, thickness)
    image = cv2.line(image, point2, point3, color2, thickness)
    image = cv2.line(image, point4, point1, color2, thickness)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(image)

def rect2points(rectangle, shape=(512,512)):

    [x, y, z, t, w] = rectangle

    # now convert this to RANGE
    # y_c = (((0.5*y)/512 -0.0442 - 0.125)/0.25)*shape[1]
    # x_c = (((0.5*x)/512 - 0.125)/0.25)*shape[0]
    x_c = x
    y_c = y
    t = np.deg2rad(90-t)
    l = 15
    # w = w/2

    l = 15
    w = w/2*(shape[0]/512)
    w=(w*(0.5/0.35)*105/100)/4
    l=l*(0.5/0.35)*1.2

    R1 = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    R2 = np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    xp_c = (np.matmul(R1, np.array([[x_c],[y_c]])))[0][0]
    yp_c = (np.matmul(R1, np.array([[x_c],[y_c]])))[1][0]

    xp_1 = xp_c + l
    yp_1 = yp_c - w
    xp_2 = xp_1
    yp_2 = yp_c + w
    xp_3 = xp_c - l
    yp_3 = yp_2
    xp_4 = xp_3
    yp_4 = yp_1

    x1 = (np.matmul(R2, np.array([[xp_1],[yp_1]])))[0][0]
    y1 = (np.matmul(R2, np.array([[xp_1],[yp_1]])))[1][0]
    x2 = (np.matmul(R2, np.array([[xp_2],[yp_2]])))[0][0]
    y2 = (np.matmul(R2, np.array([[xp_2],[yp_2]])))[1][0]
    x3 = (np.matmul(R2, np.array([[xp_3],[yp_3]])))[0][0]
    y3 = (np.matmul(R2, np.array([[xp_3],[yp_3]])))[1][0]
    x4 = (np.matmul(R2, np.array([[xp_4],[yp_4]])))[0][0]
    y4 = (np.matmul(R2, np.array([[xp_4],[yp_4]])))[1][0]

    point1 = (x1,y1)
    point2 = (x2,y2)
    point3 = (x3,y3)
    point4 = (x4,y4)

    return point1, point2, point3, point4

def points2rect(p1, p2, p3, p4, shape=(512,512)):
  
  pr = [0,0]
  pl = [0,0]
  p1 = list(p1)
  p2 = list(p2)
  p3 = list(p3)
  p4 = list(p4)
  pl[0] = (p1[0]+p4[0])/2
  pl[1] = (p1[1]+p4[1])/2
  pr[0] = (p3[0]+p2[0])/2
  pr[1] = (p3[1]+p2[1])/2
  pr = tuple(pr)
  pl = tuple(pl)

  x_r = pr[0]
  y_r = pr[1]
  x_l = pl[0]
  y_l = pl[1]

  x_c = (x_r+x_l)/2
  y_c = (y_r+y_l)/2
  thetta = np.arctan2((y_r-y_l),(x_r-x_l))/(np.pi)*180
  width = np.sqrt((y_r-y_l)**2 + (x_r-x_l)**2)

  if thetta<0:
    thetta = 360 + thetta

  rectangle = [x_c, y_c, 0, thetta, width]

  return rectangle

def camera_calibration(rectangle: list, shape=(512,512)):
    [x_c, y_c, z_c, t, width] = rectangle
    Z = z_c - 0.005
    D = 0.3302 - 0.005
    t = np.deg2rad(90-t)
    l = 15
    w = width/2
    y_c = ((0.5*y_c)/512 -0.0442 - 0.125 + 0.0079 + 0.05)/0.35*512
    x_c = ((0.5*x_c)/512 - 0.125 + 0.05)/0.35*512


    R1 = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    R2 = np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    xp_c = (np.matmul(R1, np.array([[x_c],[y_c]])))[0][0]
    yp_c = (np.matmul(R1, np.array([[x_c],[y_c]])))[1][0]

    xp_1 = xp_c + l
    yp_1 = yp_c - w
    xp_2 = xp_1
    yp_2 = yp_c + w
    xp_3 = xp_c - l
    yp_3 = yp_2
    xp_4 = xp_3
    yp_4 = yp_1

    x1 = int((np.matmul(R2, np.array([[xp_1],[yp_1]])))[0][0])
    y1 = int((np.matmul(R2, np.array([[xp_1],[yp_1]])))[1][0])
    x2 = int((np.matmul(R2, np.array([[xp_2],[yp_2]])))[0][0])
    y2 = int((np.matmul(R2, np.array([[xp_2],[yp_2]])))[1][0])
    x3 = int((np.matmul(R2, np.array([[xp_3],[yp_3]])))[0][0])
    y3 = int((np.matmul(R2, np.array([[xp_3],[yp_3]])))[1][0])
    x4 = int((np.matmul(R2, np.array([[xp_4],[yp_4]])))[0][0])
    y4 = int((np.matmul(R2, np.array([[xp_4],[yp_4]])))[1][0])

    # pr, pl = [0,0], [0,0]
    # pl[0] = (x1+x4)/2
    # pl[1] = (y1+y4)/2
    # pr[0] = (x3+x2)/2
    # pr[1] = (y3+y2)/2

    u1 = (x1-shape[0]/2)*(D)/(D-Z) + shape[0]/2
    u2 = (x2-shape[0]/2)*(D)/(D-Z) + shape[0]/2
    u3 = (x3-shape[0]/2)*(D)/(D-Z) + shape[0]/2
    u4 = (x4-shape[0]/2)*(D)/(D-Z) + shape[0]/2
    v1 = (y1-shape[1]/2)*(D)/(D-Z) + shape[1]/2
    v2 = (y2-shape[1]/2)*(D)/(D-Z) + shape[1]/2
    v3 = (y3-shape[1]/2)*(D)/(D-Z) + shape[1]/2
    v4 = (y4-shape[1]/2)*(D)/(D-Z) + shape[1]/2

    rect = points2rect((u1,v1), (u2, v2), (u3, v3), (u4, v4)) 

    return rect

def rect2maps(grasp_path: str, shape=(512,512)):
  
  with open(grasp_path,"r") as f:
        s = f.read()
  grasp = [float(s.split(",")[i]) for i in range(0,len(s.split(",")))]
  [x_c, y_c, z_c, t, width] = grasp
  grasp = camera_calibration(grasp)

  # t = np.deg2rad(90-t)
  rectangle = [x_c, y_c, z_c, t, width]
  point1, point2, point3, point4 = rect2points(rectangle, shape=shape)
  poly = Polygon([point1, point2, point3, point4])
  Q = rasterio.features.rasterize([poly], out_shape=shape)

  return Q


