from PIL import Image
import numpy as np
import time
from os import path
from os import makedirs
from random import randint
import tensorflow as tf
#from skimage.measure import compare_ssim as ssim
#from skimage.measure import compare_mse as mse2
import tensorflow.keras.backend as kb   #(pour l'erreur customisée)
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse2


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
    im_ssim = 1 - ssim (im1,im2)
    im_mse = mse(im1, im2)
    im_mse2 = mse2(im1,im2)
    print("MSE : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_mse))
    print("MSE2 : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_mse2))
    print("SSIM : " + imtemps + imradical1 + "<->" + imradical2 + " = " + str(im_ssim))


# Importer le modèle de Convolution
Temps = "2020-02-08--10-32-19--"
Model = "-c6m3"


nbjours = 50  # on peut aller jusqu'à 64 ou 65 (nombre de jours en banque de données à partir du 5 juin)
dimobjectif = (1200, 1400)
dimquimarche = (220, 220)
batchsize = 1
len_seq = 26
decalage = 28 - len_seq  # (décalage de temps entre la première image et l'image attendue)
nbcasex = 2
nbcasey = 2
long, larg = dimquimarche
nbfiltreini = 8    #( mettre une puissance de 2)


# Définition du modèle
print('Shapes du model')

tf_input = tf.keras.Input(shape=(len_seq, long, larg, 1), batch_size=batchsize)
conv_layer1 = tf.keras.layers.Conv3D(nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    tf_input)  # 218
conv_layer2 = tf.keras.layers.Conv3D(nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer1)  # 216
pool_layer1 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer2)  # 108
#bloc conv 2
conv_layer3 = tf.keras.layers.Conv3D(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    pool_layer1)  # 106
conv_layer4 = tf.keras.layers.Conv3D(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer3)  # 104
pool_layer2 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer4)  # 52
#bloc conv 3
conv_layer5 = tf.keras.layers.Conv3D(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    pool_layer2)  # 50
conv_layer6 = tf.keras.layers.Conv3D(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(
    conv_layer5)  # 48
pool_layer3 = tf.keras.layers.MaxPool3D(((1, 2, 2)))(conv_layer6)  # 23


rnn_layer1 = tf.keras.layers.ConvLSTM2D(256, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(pool_layer3)
drop1 = tf.keras.layers.Dropout(0.2) (rnn_layer1)
batch1 = tf.keras.layers.BatchNormalization(name = "batch1")(drop1)
rnn_layer2 = tf.keras.layers.ConvLSTM2D(256, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(batch1)

#bloc déconv 3
upool_layer3 = tf.keras.layers.Conv3DTranspose(4*nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(rnn_layer2) # 24 -> 50
union3 = tf.keras.layers.concatenate ([upool_layer3,conv_layer5])
deconv_layer3 = tf.keras.layers.Conv3DTranspose(4*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(union3)  # 50 -> 52
#bloc déconv 2
upool_layer2 = tf.keras.layers.Conv3DTranspose(2*nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(deconv_layer3) # 52 -> 106
union2 = tf.keras.layers.concatenate ([upool_layer2,conv_layer3])
deconv_layer2 = tf.keras.layers.Conv3DTranspose(2*nbfiltreini, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(union2)  # 106 -> 108
#bloc déconv 1
upool_layer1 = tf.keras.layers.Conv3DTranspose(nbfiltreini, (1, 4, 4), strides=(1, 2, 2), activation='relu', data_format="channels_last")(deconv_layer2) # 108 -> 218
union1 = tf.keras.layers.concatenate ([upool_layer1,conv_layer1])
deconv_layer1 = tf.keras.layers.Conv3DTranspose(1, (1, 3, 3), strides=(1, 1, 1), activation='relu', data_format="channels_last")(union1)  # 218 -> 220
tf_output = deconv_layer1


# configuration du modèle
new_model = tf.keras.Model(inputs=tf_input, outputs=tf_output)

# import des poids par transfert learning
    # Importer le modèle de Convolution
base_model = tf.keras.models.load_model('./Model-Res/'+ Temps + Model + '.h5', compile = False)
# Show the model architecture
base_model.summary()
for layer in new_model.layers:
        try:
            layer.set_weights(base_model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))
for i, layer in enumerate(new_model.layers): print(i, layer.name)
#for layer in new_model.layers[:10]: layer.trainable = False
#for layer in new_model.layers[14:]: layer.trainable = False
for layer in new_model.layers[10:15]: layer.trainable = False
print(new_model.summary())
base_model = []

# Chargement des images
t = tableau_images("./Images/Partiel/VIS8_MSG4-SEVI-MSG15-0100-NA-2019", "jpg", nbjours, 8, 9, (9, 42), (16, 28), (120,20),
                   (long, larg), nbcasex, nbcasey)   # anciennement tests au 5 juin
x_train, y_train = tableau_carré()
x_train = np.array(x_train)
y_train = np.array(y_train)
train_dim = len(x_train) - 5
print(type(x_train))
print(x_train.shape)  # tableau bidim  [nbjours * casex * casey] [ lenseq = nombre d'images de la journée ] d'images de [long] * [larg]
print(type(y_train))
print(y_train.shape)
x_train = x_train / 255.0
y_train = y_train / 255.0

print('x_train shape: ' + str(x_train.shape))
# x_train :  [nbjours * casex * casey] [ lenseq = nombre d'images de la journée ]  [long] [larg]
# x_train = np.reshape(x_train,(nbjours*nbcasex*nbcasey*len_seq,1,long,larg))
x_train = np.reshape(x_train, (nbjours * nbcasex * nbcasey, len_seq, long, larg, 1))
y_train = np.reshape(y_train, (nbjours * nbcasex * nbcasey, len_seq, long, larg, 1))

print(x_train.shape)
print(y_train.shape)


# conpilation du modèle
new_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# entrainement du modèle
new_model.fit(x_train[:train_dim], y_train[:train_dim], epochs=4, )

# Prédiction
prediction = new_model.predict(np.reshape(x_train[0], (1, len_seq, long, larg, 1)))

# Affichage
Temps = decode(prediction, "Predict-")
Temps2 = decode(y_train, Temps + "Attendu-")
Temps2 = decode(x_train, Temps + "x-")
dist_image('./Images-Res/', Temps, "Attendu-", "Attendu-", "PNG")
dist_image('./Images-Res/', Temps, "x-", "Attendu-", "PNG")
dist_image('./Images-Res/', Temps, "Predict-", "Attendu-", "PNG")

# Sauvegarde du modèle
    # sauvegarde des poids
makedirs('./Model-Res/Weights/' + Temps, exist_ok=True)
new_model.save_weights('./Model-Res/Weights/' + Temps +"/" + Temps + '-convlstm-c6m3')
    # sauvegarde du modèle entier
new_model.save('./Model-Res/' + Temps + '-convlstm-c6m3.h5')