from keras.models import load_model
from utils.losses import get_loss
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data.data_loader import get_loader
import matplotlib.pyplot as plt
import seaborn as sns

def take_tests(model_path, test_gen, path=None, figsize = (25,10), tests=[1, 3, 4]):
    """
    Evaluate your models on the validation and test sets and save the plots and images

    tests: test batches in a list. each batch with an integer >=0 and each has 15 cases for the current AppGraD.
    """

    model = load_model(model_path, custom_objects={"jaccard_loss": get_loss('jaccard_loss'),
                                                   "custom_loss": get_loss('custom_loss')})
    if path==None:
        path = model_path.split('/')
        path.pop(0)
        path.pop(-1)
        path = "/".join(path) + '/maps'
    else:
        path=path
    # model_name = model.name()

    for k in tests:
        
        obj = test_gen.__getitem__(k)[0][0][:]
        iso = test_gen.__getitem__(k)[0][1][:]
        inputs = test_gen.__getitem__(k)[0][:][:]

        y = model.predict(inputs)
        
        Q = y[:,:,:,0]
        W = y[:,:,:,1]
        S = y[:,:,:,2]
        C = y[:,:,:,3]
        
        Q_ = test_gen.__getitem__(k)[1][0][:]
        W_ = test_gen.__getitem__(k)[1][1][:]
        S_ = test_gen.__getitem__(k)[1][2][:]
        C_ = test_gen.__getitem__(k)[1][3][:]

        for i in range(0, 15):
            
            with plt.ioff():
                
                fig, axs = plt.subplots(2, 5, figsize=figsize, gridspec_kw=dict(width_ratios=[2,2,2,2,2]))
                plt.axis('off')
                axs[0,0].imshow(obj[i])
                plt.axis('off')
                axs[1,0].imshow(iso[i])
                plt.axis('off')
                sns.heatmap(Q_[i], annot=False, cmap='Reds', xticklabels=False, yticklabels=False, ax=axs[0,1], cbar=False, vmin=0, vmax=1)
                plt.axis('off')
                sns.heatmap(W_[i], annot=False, cmap='Oranges', xticklabels=False, yticklabels=False, ax=axs[0,2], cbar=False, vmin=0, vmax=1)
                plt.axis('off')
                sns.heatmap(S_[i], annot=False, cmap='Blues', xticklabels=False, yticklabels=False, ax=axs[0,3], cbar=False, vmin=0, vmax=1)
                plt.axis('off')
                sns.heatmap(C_[i], annot=False, cmap='Purples', xticklabels=False, yticklabels=False, ax=axs[0,4], cbar=False, vmin=0, vmax=1)
                plt.axis('off')
                sns.heatmap(Q[i], annot=False, cmap='Reds', xticklabels=False, yticklabels=False, ax=axs[1,1], cbar=False, vmin=0, vmax=1)
                plt.axis('off')
                sns.heatmap(W[i], annot=False, cmap='Oranges', xticklabels=False, yticklabels=False, ax=axs[1,2], cbar=False, vmin=0, vmax=1)
                plt.axis('off')
                sns.heatmap(S[i], annot=False, cmap='Blues', xticklabels=False, yticklabels=False, ax=axs[1,3], cbar=False, vmin=0, vmax=1)
                plt.axis('off')
                sns.heatmap(C[i], annot=False, cmap='Purples', xticklabels=False, yticklabels=False, ax=axs[1,4], cbar=False, vmin=0, vmax=1)
                plt.axis('off')

                plt.savefig(path+f'/test{k}_{i}.png')


# def heatmaps(model_path, test_gen, path=None, figsize = (25,10), tests=[1, 3, 4]):




