from PIL import Image
import numpy as np
import time
from os import path
from random import randint
import math
import tensorflow as tf
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse2



Temps = "2020-01-28--07-22-01--"
Model = "-c2m1"


# Recreate the exact same model, including its weights and the optimizer
base_model = tf.keras.models.load_model('./Model-Res/'+ Temps + Model + '.h5')
# Show the model architecture
base_model.summary()


for i, layer in enumerate(base_model.layers): print(i, layer.name)
for layer in base_model.layers[:4]: layer.trainable = False
for layer in base_model.layers[6:]: layer.trainable = False

nbjours = 55  # on peut aller jusqu'à 64 ou 65 (nombre de jours en banque de données à partir du 5 juin)
dimobjectif = (1200, 1400)
dimquimarche = (220, 220)
batchsize = 1
len_seq = 31
decalage = 32 - len_seq  # (décalage de temps entre la première image et l'image attendue)
nbcasex = 2
nbcasey = 2
# (57,31,40,40)

long, larg = dimquimarche
#base_model._layers[6].input = rnn_layer1

# Définition du nouveau modèle
print('Shapes du model')

tf_input = tf.keras.Input(shape=(len_seq, long, larg, 1), batch_size=batchsize)
print('input : ' + str(tf_input.shape))  # 220
conv_layer1 = tf.keras.layers.Conv3D(4, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    tf_input)  # 218
print('conv_layer1 : ' + str(conv_layer1.shape))
normac1 = tf.keras.layers.BatchNormalization()(conv_layer1)
conv_layer2 = tf.keras.layers.Conv3D(4, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    normac1)  # 216
print('conv_layer2 : ' + str(conv_layer2.shape))
#normac2 = tf.keras.layers.BatchNormalization()(conv_layer2)
pool_layer1 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer2)  # 108
print('pool_layer1 : ' + str(pool_layer1.shape))
normap1 = tf.keras.layers.BatchNormalization()(pool_layer1)

rnn_layer1 = tf.keras.layers.ConvLSTM2D(64, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(normap1)

# upool_layer1 = tf.keras.layers.UpSampling3D((1,2,2))(pool_layer1)
upool_layer1 = tf.keras.layers.Conv3DTranspose(4, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(rnn_layer1)
print('upool_layer1 : ' + str(upool_layer1.shape))
normau1 = tf.keras.layers.BatchNormalization()(upool_layer1)
deconv_layer1 = tf.keras.layers.Conv3DTranspose(1, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(normau1)
print('deconv_layer1 : ' + str(deconv_layer1.shape))
normad1 = tf.keras.layers.BatchNormalization()(deconv_layer1)
tf_output = normad1
# tf_output = tf.keras.layers.Dense(long)(deconv_layer1)
# print('output : ' + str(tf_output.shape))

# configuration du modèle
new_model = tf.keras.Model(inputs=tf_input, outputs=tf_output)

for layer in new_model.layers:
        try:
            layer.set_weights(base_model.get_layer(name=layer.name).get_weights())
            layerweights = base_model.get_layer(name=layer.name).get_weights()
            print("layer : " + layer.name)
            print (layerweights)
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

#model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)
print(new_model.summary())

for i, layer in enumerate(new_model.layers): print(i, layer.name)

'''
#tf.contrib.framework.list_variables(checkpoint_dir)
g = tf.get_default_graph()

with tf.Session() as sess:
   LoadMod = tf.train.import_meta_graph('simple_mnist.ckpt.meta')  # This object loads the model
   LoadMod.restore(sess, tf.train.latest_checkpoint('./'))# Loading weights and biases and other stuff to the model

   wc1 = g.get_tensor_by_name('wc1:0')
   sess.run( tf.assign( wc1,tf.multiply(wc1,0) ) )# Setting the values of the variable 'wc1' in the model to zero.
'''
