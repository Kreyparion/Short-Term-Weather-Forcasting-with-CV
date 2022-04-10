from PIL import Image
from matplotlib.image import imread
import numpy as np
import time
import os.path
from os import path
from random import randint
import math
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


#Fonction qui récupère toutes les images du dossier (sous réserve d'existance
#Qui les traitent directement pour ne pas les garder entières en mémoire -> trop lourd
#Qui récupère autant de petites parcelles que voulu

""" Explication rapide:
toutes les variables servent en fait à itérer l'heure pour retrouver le nom des images
Ensuite à * on a 4 fois la même chose : on récupère les nbiterx*nbitery cases
Le reste est traiter par la suite du code : on transforme les bouts d'image (cases) en numpy array et on met tout dans un grand array
On nourri finallement le RNN (Recurrent Neural Network) avec
"""

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
            if path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '41' + '.' + imtype):     #*
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

'''
Tentative infructueuse de régler le problème d'évolution de luminosité globale au cours de la journée
-> A faire quand les résultats seront assez concluants

def moyenne_couleur(im,x,y,long,larg):
    restot = 0
    for i in range(long):
        for j in range(larg):
            restot += im.getpixel((x + i, y + j))
    return restot/(long*larg)

def moyenne_des_im(tab,x,y,long,larg):
    somme = 0
    for im in tab:
        somme += moyenne_couleur(im,x,y,long,larg)
    return somme/(len(tab))
'''

def decode(prediction, label,batchsize):
    for i in range(batchsize):
        predictionres = prediction[i][29]
        predictionres = np.reshape(predictionres,(long,larg))
        #predictionres = np.vectorize(int)(predictionres*255)
        predictionres = (predictionres * 255).astype(np.uint8)
        img = Image.fromarray(predictionres, 'L')
        Labeltemps = time.strftime("%Y-%m-%d--%H-%M-%S--", time.gmtime())
        if label == "Predict-":
            img.save('./Images-Res/'+ Labeltemps + label + "n°" + str(i) + '.PNG',  format='PNG')
        else:
            img.save('./Images-Res/' +  label + "n°" + str(i) + '.PNG' , format='PNG')
        img.show()
    return Labeltemps


# Données à modifier : Voir doc pour résultats

nbjours = 53
dimobjectif = (1200,1400)
dimquimarche = (40,40)
batchsize = 1
len_seq = 31
nbcasex = 6
nbcasey = 6
#(57,31,40,40)

long,larg = dimquimarche


x_train = []
y_train = []

t = tableau_images(".\Images\VIS8_MSG4-SEVI-MSG15-0100-NA-2019", "jpg", nbjours, 8, 9, (9, 42), (17, 28), (300, 300),(long,larg),nbcasex,nbcasey)
#t = np.array(t)
#t.reshape(t,(nbcasex*nbcasey*nbjours,len_seq))



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



def melanger(l1,l2):
    n = len(l1)
    for i in range(n):
        j = randint(i,n-1)
        l1[i],l1[j] = l1[j],l1[i]
        l2[i], l2[j] = l2[j], l2[i]


melanger(x_train,y_train)                   #mélange au hasard pour que l'IA ne créée pas de pattern sur l'ordre des infos

'''
Permet de sauvegarder les données sous .txt pour pas avoir à les recalculer à chaque fois (ne marche pas encore)

fx = open("fichiersavex", "w+")
fx.write(str(x_train))
fx.close()

fy = open("fichiersavey", "w+")
fy.write(str(y_train))
fy.close()

print("2ème partie du prog")
time.sleep(10)


ffx = open("fichiersavex","r")
ffy = open("fichiersavey","r")

x_train = ffx.read()
y_train = ffy.read()
'''

x_train = np.array(x_train)
y_train = np.array(y_train)

#quelque tests de bon fonctionnement

print(type(x_train))
print(x_train.shape)
print(x_train[0])
imaa = Image.fromarray(x_train[0][5], 'L')
imaa.show()

#normalisation des données

x_train = x_train / 255.0
y_train = y_train / 255.0


# Prochaine étape avec des vecteurs vents :
# vect_vent = np.array([[(-0.04203511859914736, -0.02484401326717271), (-0.006293426597922292, 0.010194092029548093), (0.008186120297607622, 0.03806342821555688), (0.04099231333507542, -0.0017355900973617128), (0.007805028731753321, 0.07541452182114768), (-0.014492007264810524, -0.0021535412980142614), (-0.00637140050811998, 0.01746100024658214), (0.041620061984483904, 0.0005771431649331287), (-0.012319471691180202, 0.06184539627891467)], [(-0.04448620216871527, -0.007985716163116763), (-0.008332677234955572, -0.04360855399778909), (-0.008600045094645263, -0.012038711730785204), (-0.004833361688336406, -0.004715801610634192), (0.04064085353315031, -0.026221725281682023), (-0.046082798816497926, 0.015934240387437626), (-0.03302725496231839, -0.0067934748980273225), (0.006750321873098622, 0.038217230946242846), (0.02510567437245612, 0.03987347036749517)], [(-0.05595243187088291, -0.007776227674834976), (-0.01236523492697946, -0.037284538858433316), (-0.010114226023681283, -0.007452668883606369), (-0.007965756456751193, -0.010316857853268857), (0.039285163494324135, -0.009427202189283434), (-0.0747169337533985, 0.040903180248185816), (-0.04282752414987006, -0.0025238790362844088), (-0.009414848600501577, 0.06665491861202308), (-0.026270607592541264, 0.046428269580950615)], [(-0.057840860775973134, -0.05291146038134117), (-0.018135528525953265, -0.05826711215282548), (-0.009706374145144992, -0.05192631211633673), (-0.009893985733224515, -0.011414540287233875), (0.040090368227628424, -0.011016060288026588), (-0.011236026009094265, 0.0358947513476475), (-0.004998319125217184, 0.03437652044624239), (-0.059930635956777674, 0.049671514394286), (-0.06171585013494033, 0.047310205001341886)], [(-0.08063085844289901, -0.05876147343993902), (-0.05385104387341452, -0.0603821595350908), (-0.09337987397002982, -0.002533922805371939), (-0.060016474032525476, -0.002150480575749234), (-0.019097536035491202, -0.00219935690866559), (-0.008072903267548288, 0.03378721346223491), (-0.023244043903787315, 0.04368450656099961), (-0.04564710218843592, 0.039784110985244564), (-0.09429923554356208, 0.03808997121214664)], [(-0.10057736655288256, -0.015259275886134559), (-0.07465358440386656, -0.018153972405940046), (-0.05964497459372733, -0.005995379737427223), (-0.025800434947997056, -0.0012067214490646936), (-0.05487343980736815, -0.001415958454009569), (-0.045357443408384296, 0.016221351321159243), (-0.07510362947591792, 0.041418082790264064), (-0.01662250516510824, 0.03668012422224933), (-0.09230511303638216, 0.0014387517429696576)], [(-0.1053852793079244, -0.015947292970583265), (-0.1335111908003891, 0.014515430478045252), (-0.08553774908995894, -0.005742566253804662), (-0.05908449184804757, 0.01726110007758545), (-0.055221357759121285, -0.0008923073566279527), (-0.05623538398232503, 0.01897124978219343), (-0.033626910671236486, 0.01737465703343825), (-0.010666676008376423, 0.03460255703494018), (-0.0030774731004498787, 0.016521926465618248)], [(-0.10621770027280207, -0.012805052403132322), (-0.10981337832477796, -0.005476795732596902), (-0.10868606530171311, -0.0025856553020034314), (-0.06322938254306412, -0.0004758711757356573), (-0.058312358490992255, -0.0012806867376032947), (-0.05719878693535729, 0.01924985953889594), (-0.013277695484440716, 0.03474129033647178), (-0.009496302810652916, 0.01719001697047891), (-0.007239919663516049, 0.03640503390496311)], [(-0.10573546016995011, -0.014261212835578381), (-0.13032338829079418, -0.002227290554503725), (-0.1002023190277498, -3.414424470093991e-05), (-0.08203091006444273, -0.0001098404970633739), (-0.06025497237306373, 0.03378526854977603), (-0.0603895720956037, 0.036628098001808475), (-0.05769254986193081, 0.014961467355564652), (-0.027007930653655875, 0.035562997846793114), (-0.05453850623514143, 0.04289501531770934)]])


#on met les données en 1D ... Flatten ?

x_train = np.reshape(x_train,(nbjours*nbcasex*nbcasey,len_seq,long*larg))
y_train = np.reshape(y_train,(nbjours*nbcasex*nbcasey,len_seq,long*larg))

print(x_train.shape)
print(y_train.shape)
print(x_train[0])


#Structure de réseau récurrent : LSTM -LSTM - Dense - output


single_step_model = tf.keras.models.Sequential()


single_step_model.add(tf.compat.v1.keras.layers.CuDNNLSTM(2**(int(math.log(long*larg,2))-1), input_shape = (len_seq,long*larg), batch_size = batchsize, return_sequences = True, stateful = True))
single_step_model.add(tf.keras.layers.Dropout(0.2))
single_step_model.add(tf.keras.layers.BatchNormalization())     #normalise automatiquement les données en sortie


single_step_model.add(tf.compat.v1.keras.layers.CuDNNLSTM(2**(int(math.log(long*larg,2))-1), return_sequences = True, stateful = True))
single_step_model.add(tf.keras.layers.Dropout(0.1))
single_step_model.add(tf.keras.layers.BatchNormalization())


single_step_model.add(tf.keras.layers.Dense(2**(int(math.log(long*larg,2))-1), activation = 'relu'))
single_step_model.add(tf.keras.layers.Dropout(0.2))
single_step_model.add(tf.keras.layers.BatchNormalization())


single_step_model.add(tf.keras.layers.Dense((long*larg)))


single_step_model.compile(optimizer='rmsprop' ,loss='mse', metrics = ['mae'])


#entrainement du réseau de neurone

single_step_model.fit(x_train,y_train, epochs=1, )


#Un test pour vérifier le résultat

prediction = single_step_model.predict(np.reshape([x_train[i] for i in range(batchsize)],(batchsize,len_seq,long*larg)))
print(prediction*255)
Temps = decode(prediction, "Predict-",batchsize)
Temps = decode(y_train, Temps + "Attendu-",batchsize)

#Pas nécessaire (automatique à l'arrêt du programme) : permet d'être sur de vider entièrement la mémoire utilisée

y_train = []
x_train = []
single_step_model = []
