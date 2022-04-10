from PIL import Image
import numpy as np
import time
from os import path
from os import environ
from random import randint
import tensorflow as tf
import tensorflow.keras.backend as kb   #(pour l'erreur customisée)
from skimage.metrics import structural_similarity as newssim
from skimage.metrics import mean_squared_error as newmse
import matplotlib.pyplot as plt


# construction du tableau de toutes les images
def tableau_images(imname, imtype, nbjours, debmois, debjour, debut, fin, coorddeb, taille, nbiterx,
                   nbitery):  # debut, fin sont des tuples (heure,minute)
    (xdeb, ydeb), (long, larg) = coorddeb, taille
    tabres = []
    numjour = debjour - 1
    nummois = debmois
    stringmois = str(nummois)
    if nummois < 10:
        stringmois = "0" + stringmois
    for j in range(nbjours):
        print(str(j + 1) + "/" + str(nbjours))
        if (numjour == 30 and (nummois == 6)) or numjour == 31:
            numjour = 0
            nummois += 1
            stringmois = str(nummois)
            if nummois < 10:
                stringmois = "0" + stringmois
        numjour += 1
        if ((numjour == 6 and nummois == 5)  # exclusion des jours où les données ne sont pas complètes
            or (numjour == 14 and nummois == 5)
            or (numjour == 24 and nummois == 5)):
            numjour += 1
        if ((numjour == 28 and nummois == 5)  # exclusion des jours où les données ne sont pas complètes => 2 jours d'affilée
            ):
            numjour += 2
        stringjour = str(numjour)
        if numjour < 10:
            stringjour = '0' + stringjour
        res = [[[] for _ in range(nbitery)] for __ in range(nbiterx)]
        i = debut
        while i <= fin:
            (a, b) = i
            if a < 10:
                rajout = '0'
            else:
                rajout = ''
            if path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '41' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '41' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long,
                                          ydeb + (posy + 1) * larg))
                        res[posx][posy].append(imtemp)
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '44' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '44' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long,
                                          ydeb + (posy + 1) * larg))
                        res[posx][posy].append(imtemp)
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '42' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '42' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long,
                                          ydeb + (posy + 1) * larg))
                        res[posx][posy].append(imtemp)
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long,
                                          ydeb + (posy + 1) * larg))
                        res[posx][posy].append(imtemp)
            else:
                print(
                    '/!\ manque : ' + imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype)
            if b >= 57:
                i = (a + 1, 12)
            else:
                i = (a, b + 15)
        tabres.append(res)
    return tabres


# tabres est donc une structure d'images  [num jour][num case x][num case y][num image dans la journée]
# Tableau des images d'un jour donné
# de tableaux de x images par y images
# chacune ayant la même longueur et même largeur
# prises à une coordonnée de début sur l'image de la Terre
# n° de l'image dans la journée

# on fait la liste des petits carrés d'image pour l'apprentissage et pour les étiquettes (objectifs)
def tableau_carré():
    xx_train = []
    yy_train = []
    for n in range(nbcasex):
        for m in range(nbcasey):
            print("case (" + str(n) + "," + str(m) + ")")
            for i in range(nbjours):
                resx = []
                resy = []
                newim = t[i][n][m][0]
                for j in range(0, len_seq+decalage):
                    newim = t[i][n][m][j]  # [num jour][num case x][num case y][num image dans la journée]
                    if j < len_seq:
                        resx.append(np.asarray(newim))
                    if j >= decalage:
                        resy.append(np.asarray(newim))
                # newim = t[i][n][m][len_seq]
                # resy.append(np.asarray(newim))
                xx_train.append(resx)
                yy_train.append(resy)

    xx_train = np.array(xx_train)
    yy_train = np.array(yy_train)
    xx_train = xx_train / 255.0
    yy_train = yy_train / 255.0
    # print('x_train shape: ' + str(x_train.shape))
    xx_train = np.reshape(xx_train, (nbjours * nbcasex * nbcasey, len_seq, long, larg, 1))
    yy_train = np.reshape(yy_train, (nbjours * nbcasex * nbcasey, len_seq, long, larg, 1))
    print('x_train shape after reshape: ' + str(xx_train.shape))
    return xx_train, yy_train

def decode(prediction, label):
    predictionres = prediction[0][len_seq - 1]
    predictionres = np.reshape(predictionres, (long, larg))
    predictionres = (predictionres * 255).astype(np.uint8)
    img = Image.fromarray(predictionres, 'L')
    # Labeltemps = str(time.time())
    Labeltemps = time.strftime("%Y-%m-%d--%H-%M-%S--", time.gmtime())
    if label == "Predict-":
        img.save('./Images-Res/' + Labeltemps + label + '.PNG', format='PNG')
    else:
        img.save('./Images-Res/' + label + '.PNG', format='PNG')
    img.show()
    return Labeltemps

def melanger(l1, l2):
    n = len(l1)
    for i in range(n):
        j = randint(i, n - 1)
        l1[i], l1[j] = l1[j], l1[i]
        l2[i], l2[j] = l2[j], l2[i]

def custom_mse(y_actual,y_pred):                # fonction d'erreur qui sert dans les modèles neuronaux => on l'applique sur des tenseurs
    custom_loss=kb.square(10*(y_actual-y_pred))
    return custom_loss

def custom_quad(y_actual,y_pred):                # fonction d'erreur qui sert dans les modèles neuronaux => on l'applique sur des tenseurs
    custom_loss=kb.square(kb.square(6*(y_actual-y_pred)))
    return custom_loss

def custom_losses(y_actual,y_pred):                # fonction d'erreur qui sert dans les modèles neuronaux => on l'applique sur des tenseurs
    custom_loss=kb.square(kb.square(6*(y_actual-y_pred))) + kb.square(10*(y_actual-y_pred))
    return custom_loss

def mse(actual: np.ndarray, predicted: np.ndarray):      # Mean Squared Error
    return np.mean(np.square(actual - predicted))

def dist_image(impath, imtemps, imradical1, imradical2, imtype):
    im1 = Image.open(impath + imtemps + imradical1 + "." + imtype)
    im2 = Image.open(impath + imtemps + imradical2 + "." + imtype)
    im1 = np.array(im1)
    im2 = np.array(im2)
    im_mse = mse(im1, im2)
    im_new_ssim = 1 - newssim(im1,im2)
    im_new_mse = newmse(im1,im2)
    print("MSE : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_mse))
    print("MSE2 : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_new_mse))
    print("SSIM : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_new_ssim))

def plot_train_history(history, title, fichier):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))
  plt.figure()
  plt.ylim((0,3))
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.savefig ('./Images-Res/' + fichier + 'plot.png')    # sauver le fichier avant le show, car un show recrée une image vide
  plt.show()



#########
#########
#########
#########
nbjours = 120  # on peut aller jusqu'à 145 jours  (nombre de jours en banque de données du 2 mai au 30 septembre)
dimobjectif = (1200, 1400)
dimquimarche = (316, 316)
batchsize = 1
len_seq = 26
decalage = 28 - len_seq  # (décalage de temps entre la première image et l'image attendue)
nbcasex = 2
nbcasey = 4
nbfiltreini = 8#( mettre une puissance de 2)
LSTMneuronenb = 32
# (57,31,40,40)

long, larg = dimquimarche


# Définition du modèle

tf_input = tf.keras.Input(shape=(len_seq, long, larg, 1), batch_size=batchsize) #316
#bloc conv 1
conv_layer1 = tf.keras.layers.Conv3D(nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    tf_input)  # 314
conv_layer2 = tf.keras.layers.Conv3D(nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer1)  # 312
pool_layer1 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer2)  # 156
#bloc conv 2
conv_layer3 = tf.keras.layers.Conv3D(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    pool_layer1)  # 154
conv_layer4 = tf.keras.layers.Conv3D(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer3)  # 152
pool_layer2 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer4)  # 76
#bloc conv 3
conv_layer5 = tf.keras.layers.Conv3D(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    pool_layer2)  # 74
conv_layer6 = tf.keras.layers.Conv3D(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer5)  # 72
pool_layer3 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer6)  # 36
#bloc conv 4
conv_layer7 = tf.keras.layers.Conv3D(8*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    pool_layer3)  # 34
conv_layer8 = tf.keras.layers.Conv3D(8*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer7)  # 32
pool_layer4 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer8)  # 16


rnn_layer41 = tf.keras.layers.ConvLSTM2D(16*LSTMneuronenb, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(pool_layer4) #16
drop41 = tf.keras.layers.Dropout(0.2) (rnn_layer41)
batch41 = tf.keras.layers.BatchNormalization()(drop41)
rnn_layer42 = tf.keras.layers.ConvLSTM2D(16*LSTMneuronenb, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(batch41)
rnn_layer31 = tf.keras.layers.ConvLSTM2D(4*LSTMneuronenb, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(pool_layer3) #36
drop31 = tf.keras.layers.Dropout(0.2) (rnn_layer31)
batch31 = tf.keras.layers.BatchNormalization()(drop31)
rnn_layer32 = tf.keras.layers.ConvLSTM2D(4*LSTMneuronenb, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(batch31)
rnn_layer21 = tf.keras.layers.ConvLSTM2D(2*LSTMneuronenb, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(pool_layer2)
drop21 = tf.keras.layers.Dropout(0.2) (rnn_layer21)
batch21 = tf.keras.layers.BatchNormalization()(drop21)
rnn_layer22 = tf.keras.layers.ConvLSTM2D(2*LSTMneuronenb, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(batch21)
rnn_layer11 = tf.keras.layers.ConvLSTM2D(LSTMneuronenb, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(pool_layer1)
drop11 = tf.keras.layers.Dropout(0.2) (rnn_layer11)
batch11 = tf.keras.layers.BatchNormalization()(drop11)
rnn_layer12 = tf.keras.layers.ConvLSTM2D(LSTMneuronenb, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(batch11)

#bloc déconv 4
upool_layer4 = tf.keras.layers.Conv3DTranspose(8*nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(rnn_layer42) # 16 -> 34
deconv_layer4 = tf.keras.layers.Conv3DTranspose(8*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(upool_layer4)  # 34 -> 36
union4 = tf.keras.layers.concatenate ([deconv_layer4,rnn_layer32])
#bloc déconv 3
upool_layer3 = tf.keras.layers.Conv3DTranspose(4*nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(union4) # 24 -> 50
deconv_layer3 = tf.keras.layers.Conv3DTranspose(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(upool_layer3)  # 50 -> 52
union3 = tf.keras.layers.concatenate ([deconv_layer3,rnn_layer22])#bloc déconv 2
upool_layer2 = tf.keras.layers.Conv3DTranspose(2*nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(union3) # 52 -> 106
deconv_layer2 = tf.keras.layers.Conv3DTranspose(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(upool_layer2)  # 106 -> 108
union2 = tf.keras.layers.concatenate ([deconv_layer2,rnn_layer12])
#bloc déconv 1
upool_layer1 = tf.keras.layers.Conv3DTranspose(nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(union2) # 108 -> 218
union1 = tf.keras.layers.concatenate ([upool_layer1,conv_layer1])
deconv_layer1 = tf.keras.layers.Conv3DTranspose(1, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(union1)  # 218 -> 220
tf_output = deconv_layer1

# configuration du modèle
single_step_model = tf.keras.Model(inputs=tf_input, outputs=tf_output)

# conpilation du modèle
single_step_model.compile(optimizer='rmsprop', loss=custom_losses , metrics=['mae', 'mse', custom_mse, custom_quad])
# new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#single_step_model.fit(x_train[0:1], y_train[0:1], epochs=1, )
print(single_step_model.summary())


#Chargement des données
t = tableau_images("./Images/Total/VIS8_MSG4-SEVI-MSG15-0100-NA-2019", "jpg", nbjours, 5, 2, (9, 42), (17, 28), (700,1600),
                   (long, larg), nbcasex, nbcasey)   # anciennement tests au 5 juin
x_train, y_train = tableau_carré()
t = []
train_dim = (4 * len(x_train)) // 5
nbpredict = min (5, (len(x_train) // 5) -1 )
melanger(x_train,y_train)                   #mélange au hasard pour que l'IA ne créée pas de pattern sur l'ordre des infos


# entrainement du modèle
single_step_history = single_step_model.fit(x_train[:train_dim], y_train[:train_dim], epochs=25, validation_data= (x_train[train_dim:-nbpredict], y_train[train_dim:-nbpredict]), validation_steps=10)

# Prédiction
for i in range (-2,0):
    prediction = single_step_model.predict(np.reshape(x_train[i], (1, len_seq, long, larg, 1)))
    Temps = decode(prediction, "Predict-")
    Temps2 = decode(np.reshape(y_train[i], (1, len_seq, long, larg, 1)), Temps + "Attendu-")
    Temps2 = decode(np.reshape(x_train[i], (1, len_seq, long, larg, 1)), Temps + "x-")
    dist_image('./Images-Res/', Temps, "x-", "Attendu-", "PNG")
    dist_image('./Images-Res/', Temps, "Predict-", "Attendu-", "PNG")

# Sauvegarde du modèle
single_step_model.save('./Model-Res/' + Temps + '-c6m3.h5')

# Dessin de la courbe d'apprentissage
from PyQt5 import __file__ as dirname
#plugin_path = path.join("C:\Python\Miniconda3\envs\First\Library", 'plugins', 'PyQt5')
plugin_path = path.join(path.dirname(dirname), "..", "..", "..", "Library", 'plugins', 'PyQt5')
environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

plot_train_history(single_step_history,
                   Temps + 'C6M3 V3 Training and Validation loss', Temps)