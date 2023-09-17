import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import datetime
from models.BaseConvDeconv import BaseConvDeconv
from models.ResConvDeconv import ResConvDeconv
from models.VggConvDeconv import VggConvDeconv
from utils.losses import get_loss
from data.data_loader import get_loader
from keras.models import load_model
import matplotlib.pyplot as plt

MODELS = dict(BaseConvDeconv=BaseConvDeconv,
              ResConvDeconv=ResConvDeconv,
              VggConvDeconv=VggConvDeconv)

def train(batch_size=8,
          epochs = 150,
          lr = 1e-4,
          shape=(512,512,3),
          loss_name='jaccard_loss',
          model='BaseConvDeconv',
          checkpoint_path = '/content/drive/MyDrive/weights_mohokoo/checkpoints',
          resume = False,
          finalmodelpath = '/content/drive/MyDrive/weights_mohokoo/checkpoints',
          train_val_factor = 0.2,
          aug: bool=False,
          aug_p: float=0,
          val_aug_p=0,
          geo_p=0.5,
          color_p=0.5,
          noise_p=0.5,
          iso_p=0.5,
          stack: bool=True,
          min_lr = 1e-6,
          earlystop_epochs = 25,
          crop = False,
          maps=None,
          dataset_factor=1.0,
          lr_scheduler=None,
          mode='rgb'
          ):
          
          train_gen, val_gen, test_gen= get_loader(batch_size=batch_size,
                                            mode=mode,
                                            shape=shape,
                                            shuffle=True,
                                            factor=train_val_factor,
                                            aug=aug,
                                            geo_p=geo_p,
                                            color_p=color_p,
                                            noise_p=noise_p,
                                            iso_p=iso_p,
                                            aug_p=aug_p,
                                            val_aug_p=val_aug_p,
                                            stack=stack,
                                            crop=crop,
                                            maps=maps,
                                            dataset_factor=dataset_factor)
          
          if train_val_factor == 0:
            val_gen = test_gen
          
          if type(model)=='str':
              model = MODELS[model](shape=shape).get_model()
              model_name = model
          else:
              model = model
              model_name = model.name
          
          loss = get_loss(loss_name=loss_name)

          if lr_scheduler is None:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=(earlystop_epochs/2), min_lr=min_lr)
          else:
            reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

          early_stop = EarlyStopping(monitor='val_loss', patience=earlystop_epochs)
          
          if resume:
              model = load_model(checkpoint_path, 
              custom_objects={loss_name:loss}
              )
          checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          save_weights_only=False,
                                                          monitor='val_loss',
                                                          mode='min',
                                                          save_best_only=True,
                                                          initial_value_threshold=0.33
                                                          )

          model.compile(loss=loss,
                        optimizer=keras.optimizers.Adam(lr),
                        metrics = [tf.keras.metrics.MeanSquaredError(),
                                   tf.keras.metrics.RootMeanSquaredError(),
                                   tf.keras.metrics.MeanAbsoluteError(),
                                   get_loss('custom_loss'),
                                   get_loss('jaccard_loss')]
                        )
          time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
          log_dir = "/content/gag-cnn/logs/" + model_name + "/" + time
          tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
          history = model.fit(train_gen,
                              validation_data=val_gen,
                              epochs=epochs,
                              batch_size=batch_size,
                              callbacks=[early_stop, reduce_lr, checkpoint, tensorboard_callback],
                              workers=4)
          
          model.save(finalmodelpath +'/'+ model_name +'/'+ time + '/' + time + '.h5', save_format="h5")
          
          plt.plot(history.history["loss"],'r')
          plt.plot(history.history["val_loss"],'bo', markersize=2)
          plt.plot(history.history["val_loss"], 'b')
          plt.grid(color='black', linestyle='--', linewidth=1)
          fvalloss = history.history["val_loss"][-1]
          plt.title(f"final val loss is {fvalloss} for {loss_name}")
          plt.xlabel("epoch")
          plt.ylabel("loss")

          plt.savefig(finalmodelpath +'/'+ model_name +'/'+ time + '/' + time + '.png')

