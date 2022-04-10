from PIL import Image
import numpy as np
import time
from os import path
from os import makedirs
from random import randint
import tensorflow as tf
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse2


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
        res = [[[] for _ in range(nbiterx)] for __ in range(nbitery)]
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
                for j in range(0, 32):
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
    # predictionres = np.vectorize(int)(predictionres*255)
    predictionres = (predictionres * 255).astype(np.uint8)
    #print(label)
    #print(predictionres)
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


def mse(actual: np.ndarray, predicted: np.ndarray):      # Mean Squared Error
    return np.mean(np.square(actual - predicted))

def dist_image(impath, imtemps, imradical1, imradical2, imtype):
    im1 = Image.open(impath + imtemps + imradical1 + "." + imtype)
    im2 = Image.open(impath + imtemps + imradical2 + "." + imtype)
    im1 = np.array(im1)
    im2 = np.array(im2)
    im_ssim = 1 - ssim (im1,im2)
    im_mse = mse(im1, im2)
    im_mse2 = mse2(im1,im2)
    print("MSE : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_mse))
    print("MSE2 : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_mse2))
    print("SSIM : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_ssim))

#########
#########
#########
#########
nbjours = 60  # on peut aller jusqu'à 124  (nombre de jours en banque de données à partir du 30 mai)
dimobjectif = (1200, 1400)
dimquimarche = (220, 220)
batchsize = 1
len_seq = 32
decalage = 32 - len_seq  # (décalage de temps entre la première image et l'image attendue)
nbcasex = 2
nbcasey = 2
# (57,31,40,40)

long, larg = dimquimarche

# x_train = []
# y_train = []

t = tableau_images("./Images/Total/VIS8_MSG4-SEVI-MSG15-0100-NA-2019", "jpg", nbjours, 6, 5, (9, 42), (17, 28), (750,750),
                   (long, larg), nbcasex, nbcasey)
x_train, y_train = tableau_carré()

# melanger(x_train,y_train)                   #mélange au hasard pour que l'IA ne créée pas de pattern sur l'ordre des infos

x_train = np.array(x_train)
y_train = np.array(y_train)
train_dim = len(x_train) - 5

print(type(x_train))
print(
    x_train.shape)  # tableau bidim  [nbjours * casex * casey] [ lenseq = nombre d'images de la journée ] d'images de [long] * [larg]
print(x_train[0])
print(type(y_train))
print(y_train.shape)
# imaa = Image.fromarray(x_train[0][5], 'L')
# imaa.show()

x_train = x_train / 255.0
y_train = y_train / 255.0

print('x_train shape: ' + str(x_train.shape))
# x_train :  [nbjours * casex * casey] [ lenseq = nombre d'images de la journée ]  [long] [larg]
# x_train = np.reshape(x_train,(nbjours*nbcasex*nbcasey*len_seq,1,long,larg))
x_train = np.reshape(x_train, (nbjours * nbcasex * nbcasey, len_seq, long, larg, 1))
y_train = np.reshape(y_train, (nbjours * nbcasex * nbcasey, len_seq, long, larg, 1))

print(x_train.shape)
print(y_train.shape)
# print(x_train[0])


# Définition du modèle
print('Shapes du model')

tf_input = tf.keras.Input(shape=(len_seq, long, larg, 1), batch_size=batchsize)
print('input : ' + str(tf_input.shape))  # 220
#bloc conv 1
conv_layer1 = tf.keras.layers.Conv3D(4, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    tf_input)  # 218
print('conv_layer1 : ' + str(conv_layer1.shape))
normac1 = tf.keras.layers.BatchNormalization()(conv_layer1)
conv_layer2 = tf.keras.layers.Conv3D(4, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    normac1)  # 216
print('conv_layer2 : ' + str(conv_layer2.shape))
pool_layer1 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer2)  # 108
print('pool_layer1 : ' + str(pool_layer1.shape))
normap1 = tf.keras.layers.BatchNormalization()(pool_layer1)
#bloc conv 2
conv_layer3 = tf.keras.layers.Conv3D(8, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    normap1)  # 106
print('conv_layer3 : ' + str(conv_layer3.shape))
normac3 = tf.keras.layers.BatchNormalization()(conv_layer3)
conv_layer4 = tf.keras.layers.Conv3D(8, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    normac3)  # 104
print('conv_layer4 : ' + str(conv_layer4.shape))
pool_layer2 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer4)  # 52
print('pool_layer3 : ' + str(pool_layer2.shape))
normap2 = tf.keras.layers.BatchNormalization()(pool_layer2)

#bloc déconv 2
upool_layer2 = tf.keras.layers.Conv3DTranspose(8, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(normap2) # 52 -> 2014
print('upool_layer2 : ' + str(upool_layer2.shape))
normau2 = tf.keras.layers.BatchNormalization()(upool_layer2)
deconv_layer2 = tf.keras.layers.Conv3DTranspose(8, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(normau2)  # 218 -> 220
print('deconv_layer2 : ' + str(deconv_layer2.shape))
normad2 = tf.keras.layers.BatchNormalization()(deconv_layer2)
#bloc déconv 1
upool_layer1 = tf.keras.layers.Conv3DTranspose(4, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(normad2) # 108 -> 218
print('upool_layer1 : ' + str(upool_layer1.shape))
normau1 = tf.keras.layers.BatchNormalization()(upool_layer1)
deconv_layer1 = tf.keras.layers.Conv3DTranspose(1, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(normau1)  # 218 -> 220
print('deconv_layer1 : ' + str(deconv_layer1.shape))
normad1 = tf.keras.layers.BatchNormalization()(deconv_layer1)
tf_output = normad1
# tf_output = tf.keras.layers.Dense(long)(deconv_layer1)
# print('output : ' + str(tf_output.shape))

# configuration du modèle
single_step_model = tf.keras.Model(inputs=tf_input, outputs=tf_output)

# conpilation du modèle
single_step_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
single_step_model.fit(x_train[0:1], y_train[0:1], epochs=1, )
print(single_step_model.summary())

# entrainement du modèle  avec un nombre d'époque qui dépend de la convergence  (tant que l'écart varie trop d'une époque à l'autre)
ecart = 0.5
loss1 = 1
single_step_model.fit(x_train[:train_dim], y_train[:train_dim], epochs=6, )

'''
ev = new_model.evaluate(x_train[train_dim + 1:train_dim + 2], y_train[train_dim + 1:train_dim + 2],
                                batch_size=batchsize)
loss2 = int(ev[0] * 10000) / 10000
for i in range(0, 3):   # on fait entre 1 et 3 époques supplémentaires
    new_model.fit(x_train[:train_dim], y_train[:train_dim], epochs=1, )
    ev = new_model.evaluate(x_train[train_dim + 1:train_dim + 2], y_train[train_dim + 1:train_dim + 2],
                                    batch_size=batchsize)
    loss3 = int(ev[0] * 10000) / 10000
    print("loss 123 =" + str((loss1, loss2, loss3)))
    if abs(2 * loss2 - loss3 - loss1) < ecart * (loss2 - loss1):
        #break
        nothingtodo = 1
    else:
        loss1 = loss2
        loss2 = loss3


# Evaluation du modèle #
print("***** Evaluation *****")
ev = new_model.evaluate(x_train[train_dim + 1:train_dim + 2], y_train[train_dim + 1:train_dim + 2],
                                batch_size=batchsize)
print(ev)
print(ev[0])
print(ev[1])

'''

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
single_step_model.save_weights('./Model-Res/Weights/' + Temps +"/" + Temps + '-c4m2')
    # sauvegarde du modèle entier
single_step_model.save('./Model-Res/' + Temps + '-c4m2.h5')

# changements effectués
# paramètres => 25 jours pour test rapide
# ne pas mélanger
