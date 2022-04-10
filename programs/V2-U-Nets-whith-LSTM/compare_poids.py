from PIL import Image
import numpy as np
import time
from os import path
from random import randint
import math
import tensorflow as tf
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse2

#Modèle 1
Temps = "2020-01-28--07-22-01--"
Model = "-c2m1"
model1 = tf.keras.models.load_model('./Model-Res/'+ Temps + Model + '.h5')
model1.summary()

#Modèle 2
Temps = "2020-01-30--14-30-36--"
Model = "-c2m1"
model2 = tf.keras.models.load_model('./Model-Res/'+ Temps + Model + '.h5')
model2.summary()

for layer in model1.layers:
        try:
            layerweights1 = model1.get_layer(name=layer.name).get_weights()
            layerweights2 = model2.get_layer(name=layer.name).get_weights()
            print("layer : " + layer.name)
            print("Poids du modèle 1 :")
            print (layerweights1)
            print("Poids du modèle 2 :")
            print (layerweights2)
        except:
            print("pas la même couche dans les 2 modèlesr {}".format(layer.name))

