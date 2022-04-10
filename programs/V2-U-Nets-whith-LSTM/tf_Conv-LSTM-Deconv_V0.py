from PIL import Image
import numpy as np
import time
from os import path
from random import randint
import math
import tensorflow as tf


# construction du tableau de toutes les images
def tableau_images(imname,imtype,nbjours,debmois,debjour,debut,fin,coorddeb,taille,nbiterx,nbitery):        #debut, fin sont des tuples (heure,minute)
    (xdeb, ydeb), (long, larg) = coorddeb,taille
    tabres = []
    numjour = debjour-1
    nummois = debmois
    stringmois = str(nummois)
    if nummois<10:
        stringmois = "0"+stringmois
    for j in range(nbjours):
        print(str(j+1)+ "/" + str(nbjours))
        if numjour == 31:
            numjour = 0
            nummois += 1
            stringmois = str(nummois)
            if nummois < 10:
                stringmois = "0" + stringmois
        numjour += 1
        stringjour = str(numjour)
        if numjour<10:
            stringjour = '0' + stringjour
        res = [[[] for _ in range(larg)] for __ in range(long)]
        i = debut
        while i<= fin:
            (a,b) = i
            if a<10:
                rajout = '0'
            else:
                rajout = ''
            if path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '41' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '41' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long,ydeb + (posy + 1) * larg))
                        res[posx][posy].append(imtemp)
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '39' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '39' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long, ydeb + (posy + 1) * larg))
                        res[posx][posy].append(imtemp)
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '42' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '42' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long, ydeb + (posy + 1) * larg))
                        res[posx][posy].append(imtemp)
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long, ydeb + (posy + 1) * larg))
                        res[posx][posy].append(imtemp)
            else:
                print('/!\ manque : ' + imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype)
            if b >= 57:
                i = (a+1,12)
            else:
                i = (a,b+15)
        tabres.append(res)
    return tabres

# tabres est donc une structure d'images  [num jour][num case x][num case y][num image dans la journée]
# Tableau des images d'un jour donné
    # de tableaux de x images par y images
    # chacune ayant la même longueur et même largeur
    # prises à une coordonnée de début sur l'image de la Terre
    # n° de l'image dans la journée

def decode(prediction, label):
    predictionres = prediction[0][30]
    predictionres = np.reshape(predictionres,(long,larg))
    #predictionres = np.vectorize(int)(predictionres*255)
    predictionres = (predictionres * 255).astype(np.uint8)
    print (label)
    print (predictionres)
    img = Image.fromarray(predictionres, 'L')
    #Labeltemps = str(time.time())
    Labeltemps = time.strftime("%Y-%m-%d--%H-%M-%S--", time.gmtime())
    if label == "Predict-":
        img.save('./Images-Res/'+ Labeltemps + label +'.PNG',  format='PNG')
    else:
        img.save('./Images-Res/' +  label + '.PNG' , format='PNG')
    img.show()
    return Labeltemps

def melanger(l1,l2):
    n = len(l1)
    for i in range(n):
        j = randint(i,n-1)
        l1[i],l1[j] = l1[j],l1[i]
        l2[i], l2[j] = l2[j], l2[i]


nbjours = 6 # on peut aller jusqu'à 45 (nombre de jours en banque de données à partir du 09 août)
dimobjectif = (1200,1400)
dimquimarche = (46,46)
batchsize = 1
len_seq = 31
nbcasex = 2
nbcasey = 2
#(57,31,40,40)

long,larg = dimquimarche

x_train = []
y_train = []

t = tableau_images("./Images/VIS8_MSG4-SEVI-MSG15-0100-NA-2019", "jpg", nbjours, 8, 9, (9, 42), (17, 28), (300, 300),(long,larg),nbcasex,nbcasey)


# on fait la liste des petits carrés d'image pour l'apprentissage et pour les étiquettes (objectifs)
for n in range(nbcasex):
    for m in range(nbcasey):
        print("case (" + str(n)+ "," + str(m) + ")")
        for i in range(nbjours):
            resx = []
            resy = []
            newim = t[i][n][m][0]
            resx.append(np.asarray(newim))
            for j in range(1,len_seq):
                newim =t[i][n][m][j]            #[num jour][num case x][num case y][num image dans la journée]
                resx.append(np.asarray(newim))
                resy.append(np.asarray(newim))
            newim = t[i][n][m][len_seq]
            resy.append(np.asarray(newim))
            x_train.append(resx)
            y_train.append(resy)



# melanger(x_train,y_train)                   #mélange au hasard pour que l'IA ne créée pas de pattern sur l'ordre des infos

x_train = np.array(x_train)
y_train = np.array(y_train)
train_dim = len(x_train) - 5

print(type(x_train))
print(x_train.shape)  # tableau bidim  [nbjours * casex * casey] [ lenseq = nombre d'images de la journée ] d'images de [long] * [larg]
print(x_train[0])
print(type(y_train))
print(y_train.shape)
#imaa = Image.fromarray(x_train[0][5], 'L')
#imaa.show()

x_train = x_train / 255.0
y_train = y_train / 255.0



#x_train :  [nbjours * casex * casey] [ lenseq = nombre d'images de la journée ]  [long] [larg]
#x_train = np.reshape(x_train,(nbjours*nbcasex*nbcasey*len_seq,1,long,larg))
x_train = np.reshape(x_train,(nbjours*nbcasex*nbcasey, len_seq,long, larg, 1))
y_train = np.reshape(y_train,(nbjours*nbcasex*nbcasey, len_seq,long, larg, 1))

print(x_train.shape)
print(y_train.shape)
#print(x_train[0])


# Définition du modèle
print('Shapes du model')


tf_input = tf.keras.Input(shape=(len_seq,long,larg, 1),batch_size=batchsize)
print('input : ' + str (tf_input.shape))   # 220
conv_layer1 = tf.keras.layers.Conv3D(1,(1,5,5),strides=(1,1,1),  data_format = "channels_last" )(tf_input)  # 210
print('conv_layer1 : ' + str (conv_layer1.shape))
conv_layer2 = tf.keras.layers.Conv3D(2,(1,11,11),strides=(1,1,1),  data_format = "channels_last" )(conv_layer1) # 68
print('conv_layer2 : ' + str (conv_layer2.shape))
conv_layer3 = tf.keras.layers.Conv3D(4,(1,11,11),strides=(1,1,1),  data_format = "channels_last" )(conv_layer2)  # 32
print('conv_layer3 : ' + str (conv_layer3.shape))
conv_layer4 = tf.keras.layers.Conv3D(4,(1,5,5),strides=(1,1,1),  data_format = "channels_last" )(conv_layer3)  # 32
print('conv_layer4 : ' + str (conv_layer4.shape))
conv_layer5 = tf.keras.layers.Conv3D(4,(1,3,3),strides=(1,1,1),  data_format = "channels_last" )(conv_layer4)  # 32
print('conv_layer5 : ' + str (conv_layer5.shape))
rnn_input = tf.reshape(conv_layer5, (1,31,16*16*4))
print('rnn_input : ' + str (rnn_input.shape))
rnn_layer1 = tf.keras.layers.LSTM(1024,  return_sequences=True, stateful=True)(rnn_input)
print( 'rnn_layer1 : ' + str(rnn_layer1.shape))
droupout1 = tf.keras.layers.Dropout(0.1)(rnn_layer1)  # on drope certains neurones
norma1 = tf.keras.layers.BatchNormalization()(droupout1)
rnn_layer2 = tf.keras.layers.LSTM(1024,  return_sequences=True, stateful=True)(norma1)
print( 'rnn_layer2 : ' + str(rnn_layer2.shape))
droupout2 = tf.keras.layers.Dropout(0.1)(rnn_layer2)  # on drope certains neurones
norma2 = tf.keras.layers.BatchNormalization()(droupout2)
#rnn_layer2 = tf.keras.layers.ConvLSTM2D(64, (1,1),  return_sequences=True, stateful=True, data_format = "channels_last")(norma1)
#print( 'rnn_layer2 : ' + str(rnn_layer2.shape))
#droupout2 = tf.keras.layers.Dropout(0.1)(rnn_layer2)
#norma2 = tf.keras.layers.BatchNormalization()(droupout2)
#Upool_layer2 = tf.keras.layers.UpSampling3D (norma2)
#print('Upool_layer2 : ' + str (Upool_layer2.shape))
rnn_output = tf.reshape(norma2, (1,31,16,16,4))
print('rnn_output : ' + str (rnn_output.shape))
deconv_layer5 = tf.keras.layers.Conv3DTranspose(64,(1,3,3),strides=(1,1,1),  data_format = "channels_last" )(rnn_output)
print('deconv_layer5 : ' + str (deconv_layer5.shape))
deconv_layer4 = tf.keras.layers.Conv3DTranspose(32,(1,5,5),strides=(1,1,1),  data_format = "channels_last" )(deconv_layer5)
print('deconv_layer4 : ' + str (deconv_layer4.shape))
deconv_layer3 = tf.keras.layers.Conv3DTranspose(8,(1,11,11),strides=(1,1,1),  data_format = "channels_last" )(deconv_layer4)
print('deconv_layer3 : ' + str (deconv_layer3.shape))
deconv_layer2 = tf.keras.layers.Conv3DTranspose(2,(1,11,11),strides=(1,1,1), data_format = "channels_last" )(deconv_layer3)
print('deconv_layer2 : ' + str (deconv_layer2.shape))
#Upool_layer1 = tf.keras.layers.UpSampling3D(deconv_layer2)
#print('Upool_layer1 : ' + str(Upool_layer2.shape))
deconv_layer1 = tf.keras.layers.Conv3DTranspose(1,(1,5,5),strides=(1,1,1), data_format = "channels_last")(deconv_layer2)
print('deconv_layer1 : ' + str (deconv_layer1.shape))
tf_output = deconv_layer1
#tf_output = tf.keras.layers.Dense(long)(deconv_layer1)
#print('output : ' + str(tf_output.shape))

# configuration du modèle
single_step_model = tf.keras.Model(inputs = tf_input, outputs = tf_output)

# conpilation du modèle
single_step_model.compile(optimizer='rmsprop' ,loss='mse', metrics = ['mae'])
#new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

single_step_model.fit(x_train[0:1],y_train[0:1], epochs=1, )
print(single_step_model.summary())
# entrainement du modèle
single_step_model.fit(x_train[:train_dim],y_train[:train_dim], epochs=2, )

#Prédiction
prediction = single_step_model.predict(np.reshape(x_train[0],(1,len_seq, long, larg, 1)))
print(prediction*255)

#Affichage
#prediction = np.reshape (prediction, (len_seq)
Temps = decode(prediction *2, "Predict-")
Temps = decode(y_train, Temps + "Attendu-")

'''
new_model.fit(x_train,y_train, epochs=1, )
new_model.fit(x_train,y_train, epochs=1, )
#prediction = new_model.predict(np.reshape(x_train[train_dim + 1],(1,len_seq,long*larg)))
prediction = new_model.predict(np.reshape(x_train[0],(1,len_seq,long*larg)))
print(prediction*255)

Temps = decode(prediction, "Predict-")
Temps = decode(y_train, Temps + "Attendu-")

'''
# changements effectués
# paramètres => 25 jours pour test rapide
# ne pas mélanger