from PIL import Image
import numpy as np
import time
from os import path
from os import makedirs
from random import randint
import tensorflow as tf
import tensorflow.keras.backend as kb   #(pour l'erreur customisée)
from skimage.metrics import structural_similarity as newssim
from skimage.metrics import mean_squared_error as newmse


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
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '39' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '39' + '.' + imtype)
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

#########
#########
#########
#########
nbjours = 50  # on peut aller jusqu'à 52 jours  (nombre de jours en banque de données du 9 août au 30 septembre)
dimobjectif = (1200, 1400)
dimquimarche = (220, 220)
batchsize = 1
len_seq = 26
decalage = 28 - len_seq  # (décalage de temps entre la première image et l'image attendue)
nbcasex = 1
nbcasey = 4
nbfiltreini = 4    #( mettre une puissance de 2)
# (57,31,40,40)

long, larg = dimquimarche


# Définition du modèle
print('Shapes du model')

tf_input = tf.keras.Input(shape=(len_seq, long, larg, 1), batch_size=batchsize)
#print('input : ' + str(tf_input.shape))  # 220
#bloc conv 1
conv_layer1 = tf.keras.layers.Conv3D(nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    tf_input)  # 218
#print('conv_layer1 : ' + str(conv_layer1.shape))
#normac1 = tf.keras.layers.BatchNormalization()(conv_layer1)
conv_layer2 = tf.keras.layers.Conv3D(nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer1)  # 216
#print('conv_layer2 : ' + str(conv_layer2.shape))
pool_layer1 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer2)  # 108
#print('pool_layer1 : ' + str(pool_layer1.shape))
#normap1 = tf.keras.layers.BatchNormalization()(pool_layer1)
#bloc conv 2
conv_layer3 = tf.keras.layers.Conv3D(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    pool_layer1)  # 106
#print('conv_layer3 : ' + str(conv_layer3.shape))
#normac3 = tf.keras.layers.BatchNormalization()(conv_layer3)
conv_layer4 = tf.keras.layers.Conv3D(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer3)  # 104
#print('conv_layer4 : ' + str(conv_layer4.shape))
pool_layer2 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer4)  # 52
#print('pool_layer3 : ' + str(pool_layer2.shape))
#normap2 = tf.keras.layers.BatchNormalization()(pool_layer2)
#bloc conv 3
conv_layer5 = tf.keras.layers.Conv3D(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    pool_layer2)  # 50
#print('conv_layer5 : ' + str(conv_layer5.shape))
#normac4 = tf.keras.layers.BatchNormalization()(conv_layer5)
conv_layer6 = tf.keras.layers.Conv3D(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer5)  # 48
#print('conv_layer6 : ' + str(conv_layer6.shape))
#normac5 = tf.keras.layers.BatchNormalization()(conv_layer6)
#conv_layer7 = tf.keras.layers.Conv3D(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(conv_layer6)  # 46
#print('conv_layer7 : ' + str(conv_layer7.shape))
pool_layer3 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer6)  # 23
#print('pool_layer3 : ' + str(pool_layer3.shape))
#normap3 = tf.keras.layers.BatchNormalization()(pool_layer3)


rnn_layer1 = tf.keras.layers.ConvLSTM2D(256, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(pool_layer3)
drop1 = tf.keras.layers.Dropout(0.2) (rnn_layer1)
batch1 = tf.keras.layers.BatchNormalization(name = "batch1")(drop1)
rnn_layer2 = tf.keras.layers.ConvLSTM2D(256, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(batch1)

#bloc déconv 3
upool_layer3 = tf.keras.layers.Conv3DTranspose(4*nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(rnn_layer2) # 24 -> 50
#print('upool_layer3 : ' + str(upool_layer3.shape))
#normau3 = tf.keras.layers.BatchNormalization()(upool_layer3)
#deconv_layer4 = tf.keras.layers.Conv3DTranspose(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(upool_layer3)  # 50 -> 52
#print('deconv_layer4 : ' + str(deconv_layer4.shape))
#normad3 = tf.keras.layers.BatchNormalization()(deconv_layer4)
union3 = tf.keras.layers.concatenate ([upool_layer3,conv_layer5])
deconv_layer3 = tf.keras.layers.Conv3DTranspose(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(union3)  # 50 -> 52
#print('deconv_layer3 : ' + str(deconv_layer3.shape))
#normad3 = tf.keras.layers.BatchNormalization()(deconv_layer3)
#bloc déconv 2
upool_layer2 = tf.keras.layers.Conv3DTranspose(2*nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(deconv_layer3) # 52 -> 106
#print('upool_layer2 : ' + str(upool_layer2.shape))
#normau2 = tf.keras.layers.BatchNormalization()(upool_layer2)
union2 = tf.keras.layers.concatenate ([upool_layer2,conv_layer3])
deconv_layer2 = tf.keras.layers.Conv3DTranspose(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(union2)  # 106 -> 108
#print('deconv_layer2 : ' + str(deconv_layer2.shape))
#normad2 = tf.keras.layers.BatchNormalization()(deconv_layer2)
#bloc déconv 1
upool_layer1 = tf.keras.layers.Conv3DTranspose(nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(deconv_layer2) # 108 -> 218
#print('upool_layer1 : ' + str(upool_layer1.shape))
#normau1 = tf.keras.layers.BatchNormalization()(upool_layer1)
union1 = tf.keras.layers.concatenate ([upool_layer1,conv_layer1])
deconv_layer1 = tf.keras.layers.Conv3DTranspose(1, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(union1)  # 218 -> 220
#print('deconv_layer1 : ' + str(deconv_layer1.shape))
#normad1 = tf.keras.layers.BatchNormalization()(deconv_layer1)
tf_output = deconv_layer1
# tf_output = tf.keras.layers.Dense(long)(deconv_layer1)
# print('output : ' + str(tf_output.shape))

# configuration du modèle
single_step_model = tf.keras.Model(inputs=tf_input, outputs=tf_output)

# conpilation du modèle
single_step_model.compile(optimizer='rmsprop', loss=custom_losses , metrics=['mae', 'mse', custom_mse, custom_quad])
# new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#single_step_model.fit(x_train[0:1], y_train[0:1], epochs=1, )
print(single_step_model.summary())


#Chargement des données
t = tableau_images("./Images/Partiel/VIS8_MSG4-SEVI-MSG15-0100-NA-2019", "jpg", nbjours, 8, 9, (9, 42), (17, 28), (120,20),
                   (long, larg), nbcasex, nbcasey)   # anciennement tests au 5 juin
x_train, y_train = tableau_carré()
t = []
# melanger(x_train,y_train)                   #mélange au hasard pour que l'IA ne créée pas de pattern sur l'ordre des infos
x_train = np.array(x_train)
y_train = np.array(y_train)
train_dim = len(x_train) - 5
x_train = x_train / 255.0
y_train = y_train / 255.0
print('x_train shape: ' + str(x_train.shape))
x_train = np.reshape(x_train, (nbjours * nbcasex * nbcasey, len_seq, long, larg, 1))
y_train = np.reshape(y_train, (nbjours * nbcasex * nbcasey, len_seq, long, larg, 1))
print('x_train shape after reshape: ' + str(x_train.shape))


# entrainement du modèle
single_step_model.fit(x_train[:train_dim], y_train[:train_dim], epochs=8, )

# Prédiction
prediction = single_step_model.predict(np.reshape(x_train[0], (1, len_seq, long, larg, 1)))
#print(prediction * 255)

# Affichage
# prediction = np.reshape (prediction, (len_seq)
Temps = decode(prediction, "Predict-")
Temps2 = decode(y_train, Temps + "Attendu-")
Temps2 = decode(x_train, Temps + "x-")
dist_image('./Images-Res/', Temps, "Attendu-", "Attendu-", "PNG")
dist_image('./Images-Res/', Temps, "x-", "Attendu-", "PNG")
dist_image('./Images-Res/', Temps, "Predict-", "Attendu-", "PNG")

# Sauvegarde du modèle
    # sauvegarde des poids
makedirs('./Model-Res/Weights/' + Temps, exist_ok=True)
single_step_model.save_weights('./Model-Res/Weights/' + Temps +"/" + Temps + '-c6m3')
    # sauvegarde du modèle entier
single_step_model.save('./Model-Res/' + Temps + '-c6m3.h5')

# changements effectués
# paramètres => 25 jours pour test rapide
# ne pas mélanger
