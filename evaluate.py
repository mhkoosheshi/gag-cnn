from keras.models import load_model
from utils.losses import get_loss
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data.data_loader import get_loader
import matplotlib.pyplot as plt
import seaborn as sns

def take_tests(model_path, test_gen, figsize = (25,10), tests=[1, 3, 4]):
    """
    Evaluate your models on the validation and test sets and save the plots and images

    tests: test batches in a list. each batch with an integer >=0 and each has 15 cases for the current AppGraD.
    """
    plt.ioff()
    model = load_model(model_path)
    path = (model_path.split('/').pop())
    path = "/".join(path) + '/maps'
    model_name = model.name()

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

        for i in range(0, test_gen.batch_size()):
            
            with plt.ioff():
                # add correct captions
                fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 1)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                plt.imshow(obj[i])

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 2)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                sns.heatmap(Q_[i], cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=np.max(Q_[i]))

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 3)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                sns.heatmap(W_[i], cmap='Oranges', xticklabels=False, yticklabels=False, vmin=0, vmax=np.max(W_[i]))

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 4)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                sns.heatmap(S_[i], cmap='Blues', xticklabels=False, yticklabels=False, vmin=0, vmax=np.max(S_[i]))

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 5)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                sns.heatmap(C_[i], cmap='Purples', xticklabels=False, yticklabels=False, vmin=0, vmax=np.max(C_[i]))

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 6)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                plt.imshow(iso[i])

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 7)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                sns.heatmap(Q[i], cmap='Reds', xticklabels=False, yticklabels=False, vmin=0, vmax=np.max(Q[i]))

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 8)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                sns.heatmap(W[i], cmap='Oranges', xticklabels=False, yticklabels=False, vmin=0, vmax=np.max(W[i]))

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 9)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                sns.heatmap(S[i], cmap='Blues', xticklabels=False, yticklabels=False, vmin=0, vmax=np.max(S[i]))

                # fig = plt.figure(figsize=figsize)
                fig.add_subplot(2, 5, 10)
                plt.axis('off')
                string = ''
                plt.title(f"{string}")
                sns.heatmap(C[i], cmap='Purples', xticklabels=False, yticklabels=False, vmin=0, vmax=np.max(C[i]))

                plt.savefig(path+f'/test{k}{i}.png')



