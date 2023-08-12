import cv2
from utils import draw_grasp
import matplotlib.pyplot as plt
import numpy as np

image_path = r'C:\Users\lenovo\Desktop\simulation\stage 3 - python\records\obj\rgb1230.png'
grasp_path = r'C:\Users\lenovo\Desktop\simulation\stage 3 - python\records\grasp\grasp1230.txt'

draw_grasp(image_path, grasp_path)

