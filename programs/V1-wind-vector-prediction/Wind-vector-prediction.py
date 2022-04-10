from PIL import Image
import numpy as np
import time
from os import path
from os import environ
from os import makedirs
from random import randint
import tensorflow as tf
import tensorflow.keras.backend as kb   #(pour l'erreur customisée)
#from skimage.metrics import structural_similarity as newssim
#from skimage.metrics import mean_squared_error as newmse
from skimage.measure import compare_ssim as newssim
from skimage.measure import compare_mse as newmse
import matplotlib.pyplot as plt
import math

def tableau_vecteurs(txtfilename):
    global pas,ydeb,xdeb
    f=open(txtfilename, "r")

    res = f.read()
    print(res)

    res = res.split('fin_info')

    pas = int(res.pop())
    ydeb = int(res.pop())
    xdeb = int(res.pop())

    res = res[0]

    res = res.split('fin_jour')
    res.pop()
    for i in range(len(res)):
        res[i] = res[i].split('fin_casey')
        res[i].pop()

    print(len(res),len(res[0]))

    for i in range(len(res)):
        for j in range(len(res[0])):
            res[i][j] = res[i][j].split('fin_casex')
            res[i][j].pop()

    print(len(res),len(res[0]),len(res[0][0]))

    for i in range(len(res)):
        for j in range(len(res[0])):
            for k in range(len(res[0][0])):
                res[i][j][k] = res[i][j][k].split('fin_im')
                res[i][j][k].pop()

    print(res)
    print(len(res),len(res[0]),len(res[0][0]),len(res[0][0][0]))

    for i in range(len(res)):
        for j in range(len(res[0])):
            for k in range(len(res[0][0])):
                for l in range(len(res[0][0][0])):
                    res[i][j][k][l] = res[i][j][k][l].split('fin_ligne')
                    res[i][j][k][l].pop()

    for i in range(len(res)):
        for j in range(len(res[0])):
            for k in range(len(res[0][0])):
                for l in range(len(res[0][0][0])):
                    for m in range(len(res[0][0][0][0])):
                        res[i][j][k][l][m] = res[i][j][k][l][m].split('fin_coord')
                        res[i][j][k][l][m].pop()

    for i in range(len(res)):
        for j in range(len(res[0])):
            for k in range(len(res[0][0])):
                for l in range(len(res[0][0][0])):
                    for m in range(len(res[0][0][0][0])):
                        for n in range(len(res[0][0][0][0][0])):
                            res[i][j][k][l][m][n] = res[i][j][k][l][m][n].split('fin_val')
                            res[i][j][k][l][m][n] = [float(res[i][j][k][l][m][n][0]),float(res[i][j][k][l][m][n][1])]

    print(res)
    print(len(res),len(res[0]),len(res[0][0]),len(res[0][0][0]),len(res[0][0][0][0]),len(res[0][0][0][0][0]),len(res[0][0][0][0][0][0]))

    return res


def tableau_carré():
    xx_train = []
    yy_train = []
    for n in range(nbcasey):
        for m in range(nbcasex):
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
    predictionres = np.reshape(predictionres, (long, larg,nb_info))
    X, Y, U, V = [], [], [], []
    for i, a in enumerate(predictionres):
        for j, (vx, vy) in enumerate(a):
            X.append(xdeb+i*pas)
            Y.append(ydeb-j*pas)
            U.append(-vy)
            V.append(vx)
    plt.figure()
    plt.quiver(X, Y, U, V)

    # Labeltemps = str(time.time())
    Labeltemps = time.strftime("%Y-%m-%d--%H-%M-%S--", time.gmtime())
    plt.title(Labeltemps + label)
    plt.legend()
    if label == "Predict-":
        plt.savefig('./Vect-Res/' + Labeltemps + label + 'plot.png')  # sauver le fichier avant le show, car un show recrée une image vide
    else:
        plt.savefig('./Vect-Res/' + label + 'plot.png')
    plt.show()
    return Labeltemps,predictionres

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

def dist_vect(vecttemps, vect1, vect2, vectradical1, vectradical2):
    vect1 = np.array(vect1)
    vect2 = np.array(vect2)
    print(vect1.shape)
    im_mse = mse(vect1, vect2)
    #im_new_ssim = 1 - newssim(vect1,vect2)
    im_new_mse = newmse(vect1,vect2)
    print("MSE : " + vecttemps + vectradical1 + "<->" + vectradical2 + " = " + str(im_mse))
    print("MSE2 : " + vecttemps + vectradical1 + "<->" + vectradical2 + " = " + str(im_new_mse))
    #print("SSIM : " + vecttemps + vectradical1 + "<->" + vectradical2 + " = " + str(im_new_ssim))

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


  plt.savefig ('./Vect-Res/' + fichier + 'plot.png')    # sauver le fichier avant le show, car un show recrée une image vide
  plt.show()


#Chargement des données
t = tableau_vecteurs("tab_vents4.0.txt") + tableau_vecteurs("tab_vents4.1.txt") + tableau_vecteurs("tab_vents4.2.txt") + tableau_vecteurs("tab_vents4.3.txt") + tableau_vecteurs("tab_vents4.4.txt") + tableau_vecteurs("tab_vents4.5.txt") + tableau_vecteurs("tab_vents4.6.txt") + tableau_vecteurs("tab_vents4.7.txt")

#t = tableau_vecteurs("tab_vents3.testtt.txt")

nbjours = len(t)  # on peut aller jusqu'à 64 ou 65 (nombre de jours en banque de données à partir du 5 juin)
dimobjectif = (20,20)
dimquimarche = (len(t[0][0][0][0]), len(t[0][0][0][0][0]))
batchsize = 1
len_dispo = len(t[0][0][0])
decalage = 2            # (décalage de temps entre la première image et l'image attendue)
len_seq = len_dispo - decalage
nbcasex = len(t[0][0])
nbcasey = len(t[0])
long, larg = dimquimarche
Epochs = 25
nb_info = len(t[0][0][0][0][0][0])
# (57,31,40,40)


x_train, y_train = tableau_carré()
t = []

melanger(x_train,y_train)                   #mélange au hasard pour que l'IA ne créée pas de pattern sur l'ordre des infos

train_dim = (3 * len(x_train)) // 4
nbpredict = min (5, (len(x_train) // 4) -1 )
print('nb de cas utilisés pour la prédiction : ' + str(nbpredict))

#on met les données en 1D ... Flatten ?

x_train = np.reshape(x_train,(nbjours*nbcasex*nbcasey,len_seq,long*larg*nb_info))
y_train = np.reshape(y_train,(nbjours*nbcasex*nbcasey,len_seq,long*larg*nb_info))


x_valid, y_valid = x_train[train_dim:-nbpredict], y_train[train_dim:-nbpredict]
x_train, y_train = x_train[:train_dim], y_train[:train_dim]


print(x_train.shape)
print(y_train.shape)
print(x_train[0][0])

# Définition du modèle
print('Shapes du model')

single_step_model = tf.keras.models.Sequential()

single_step_model.add(tf.compat.v1.keras.layers.CuDNNLSTM(2**(int(math.log(long*larg*nb_info,2))), input_shape = (len_seq,long*larg*nb_info), batch_size = batchsize, return_sequences = True, stateful = True))
single_step_model.add(tf.keras.layers.Dropout(0.2))
single_step_model.add(tf.keras.layers.BatchNormalization())     #normalise automatiquement les données en sortie


single_step_model.add(tf.compat.v1.keras.layers.CuDNNLSTM(2**(int(math.log(long*larg*nb_info,2))), return_sequences = True, stateful = True))
single_step_model.add(tf.keras.layers.Dropout(0.1))
single_step_model.add(tf.keras.layers.BatchNormalization())


single_step_model.add(tf.keras.layers.Dense((long*larg*nb_info)))


# conpilation du modèle
single_step_model.compile(optimizer='rmsprop', loss=custom_losses , metrics=['mae', 'mse', custom_mse, custom_quad])
print(single_step_model.summary())





# entrainement du modèle
single_step_history = single_step_model.fit(x_train, y_train, epochs=Epochs, validation_data= (x_valid, y_valid), validation_steps=10)

# Prédiction
for i in range (-2,0):
    prediction = single_step_model.predict(np.reshape(x_train[i], (1,len_seq,long*larg*nb_info)))
    #print(prediction * 255)

    # Affichage
    # prediction = np.reshape (prediction, (len_seq)
    Temps,predict_vect = decode(prediction, "Predict-")
    Temps2,attendu_vect = decode(np.reshape(y_train[i], (1,len_seq,long*larg*nb_info)), Temps + "Attendu-")
    Temps2,x_vect = decode(np.reshape(x_train[i], (1,len_seq,long*larg*nb_info)), Temps + "x-")
    #dist_image('./Images-Res/', Temps, "Attendu-", "Attendu-", "PNG")
    dist_vect(Temps, x_vect, attendu_vect, "x-" , "Attendu-")
    dist_vect(Temps, predict_vect, attendu_vect, "Predict-", "Attendu-")

# Sauvegarde du modèle
    # sauvegarde des poids
    #makedirs('./Model-Res/Weights/' + Temps, exist_ok=True)
    #single_step_model.save_weights('./Model-Res/Weights/' + Temps +"/" + Temps + '-c6m3')
    # sauvegarde du modèle entier
single_step_model.save('./Model-Res-Vect/' + Temps + '-3lstm.h5')


# Dessin de la courbe d'apprentissage
from PyQt5 import __file__ as dirname
#plugin_path = path.join("C:\Python\Miniconda3\envs\First\Library", 'plugins', 'PyQt5')

#plugin_path = paC:\Dev\Python\Lib\site-packages\PyQt5\Qt\plugins\platforms
print(dirname)
plugin_path = path.join(path.dirname(dirname), "Qt", 'plugins', 'platforms')
environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

plot_train_history(single_step_history,
                   Temps + 'C6M3 V2 Training and Validation loss', Temps)



"""
for loop in range(len(TAB)):
    X,Y,U,V = [],[],[],[]
    for i, a in enumerate(TAB[loop]):
        for j, (x, y, vx, vy) in enumerate(a):
            X.append(x)
            Y.append(y)
            U.append(-vx)
            V.append(vy)
    plt.figure()

    plt.quiver(X,Y,U,V)
plt.show()
"""
