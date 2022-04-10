from PIL import Image
import numpy as np
import time
import os.path
from os import path
import matplotlib.pyplot as plt
#im1.show()  #4.5 km par pixel (avec l'espagne)
#im2.show()

"""
def soustraction(im1,im2):      #soustraction de deux images
    if im1.size != im2.size:
        return False
    (long,larg) = im1.size
    imres = Image.new("L", (long, larg), "black")
    for i in range(long):
        for j in range(larg):
            imres.putpixel((i, j),abs(im1.getpixel((i,j))-im2.getpixel((i,j))))
    return imres
"""


def moyenne_couleur(im,x,y,long,larg):
    restot = 0
    for i in range(long):
        for j in range(larg):
            restot += im.getpixel((x + i, y + j))
    return restot/(long*larg)

def ecart_moyen_couleur(im1,x1,y1,im2,x2,y2,long,larg,moy1):    # corrélation entre 2 carrés de l'images, complexité O(long*larg)
    restot = 0
    moy2 = moyenne_couleur(im2,x2,y2,long,larg)
    for i in range(long):
        for j in range(larg):
            restot += ((im1.getpixel((x1+i,y1+j)) - moy1) - (im2.getpixel((x2+i,y2+j)) - moy2))**2      #on enlève la moyenne : si une partie s'est obsurcie / éclairée : superposition de nuages
    return (restot/(long*larg))

def mse(actual: np.ndarray, predicted: np.ndarray):      # Mean Squared Error
    return np.mean(np.square(actual - predicted))

def ecart_moyen_couleur2(im1_array,x1,y1,im2_array,x2,y2,long,larg,moy1):
    if len(im2_array[x2:(x2+long), y2:(y2+larg)]) == 0 or len(im2_array[x2:(x2+long), y2:(y2+larg)][0]) == 0:
        print (x2,y2,x1,y1)
        return 1000000
    return mse(im1_array[x1:(x1+long), y1:(y1+larg)]-moy1, im2_array[x2:(x2+long), y2:(y2+larg)]-np.mean(im2_array[x2:(x2+long), y2:(y2+larg)]))

#soustraction(im1,im2).show()

def ecart_type(im,x,y,long,larg):
    restot = 0
    moy = moyenne_couleur(im,x,y,long,larg)
    for i in range(long):
        for j in range(larg):
            restot += (im.getpixel((x + i, y + j)) - moy)**2
    return np.sqrt(restot / (long * larg))


def trouver_pos_newimage(im1_array,x1,y1,long1,larg1,im2_array,x2,y2,long2,larg2):  # O((long2-long1)*(larg2-larg1)*long*larg)
    mini = 100000000
    posx = -1
    posy = -1
    #moy1 = np.mean(im1_array)    ####################"""  restreint à la largeur
    moy1 = np.mean(im1_array[x1:(x1+long), y1:(y1+larg)])
    for i in range(long2-long1):
        for j in range(larg2-larg1):
            ecart = ecart_moyen_couleur2(im1_array, x1, y1, im2_array, x2+i, y2+j, long1, larg1,moy1)
            if ecart < mini:
                posx = x2+i
                posy = y2+j
                mini = ecart

    valxpre = ecart_moyen_couleur2(im1_array, x1, y1, im2_array, posx-1, posy, long1, larg1,moy1)
    valxsuiv = ecart_moyen_couleur2(im1_array, x1, y1, im2_array, posx + 1, posy, long1, larg1, moy1)
    valypre = ecart_moyen_couleur2(im1_array, x1, y1, im2_array, posx, posy - 1, long1, larg1, moy1)
    valysuiv = ecart_moyen_couleur2(im1_array, x1, y1, im2_array, posx, posy + 1, long1, larg1, moy1)
    #print (mini, valxpre,valxsuiv,valypre,valysuiv)
    if valxsuiv == valxpre:
        epsilonx = 0
    elif valxsuiv == mini:
        epsilonx = 0.4999
    elif valxpre == mini:
        epsilonx = -0.4999
    else:
        #epsilonx = np.arctan(mini/(valxsuiv-mini)-mini/(valxpre-mini))/np.pi
        epsilonx = (valxpre - valxsuiv) / (4 *( valxsuiv + valxpre) - 2 * mini)
    if valysuiv == valypre:
        epsilony = 0
    elif valysuiv == mini:
        epsilony = 0.4999
    elif valypre == mini:
        epsilony = -0.4999
    else:
        #epsilony = np.arctan(mini / (valysuiv - mini) - mini / (valypre - mini)) / np.pi
        epsilony = (valypre - valysuiv) / (4 *( valysuiv + valypre) - 2 * mini)
    return (posx+epsilonx,posy+epsilony)


"""
def afficher_rogné(im,x,y,long,larg):
    im.crop((x, y, x+long, y+larg)).show()
"""


#afficher_rogné(im1,500,530,20,20)
#afficher_rogné(im2,470,500,60,60)

#a = trouver_pos_newimage(im1,500,530,20,20,im2,470,500,60,60)
#afficher_rogné(im2,x,y,20,20)
#print(a)

"""
def afficher_vents(liste,im):
    for (x,y,vx,vy) in liste:
        norme = int(np.sqrt(vx*vx+vy*vy)*4)
        try:
            im.putpixel((int(x), int(y)), (255,255,255))
        except:
            "rien"
        for loop in range(1,norme):
            try:
                im.putpixel((int(x+vx*loop/3),int(y+vy*loop/3)),(230,30,30))
            except:
                "rien"
    im.show()
"""

def grand_tableau(im1,im2,im3,im4,coorddeb,taille,temps,pas):
    x,y = coorddeb
    long,larg = taille
    encadre = 7  #le déplacement maximum est fixé à 0.5*pas=7 pixels en temps=15min  Or 1 pixel ~ 3km donc des vents de < 84km/h
    cadre = 16  #cadre d'étude du pattern à retrouver
    nbsurligne = int((long - 6* encadre -cadre -8)/pas)
    nbsurcollone = int((long - 6* encadre -cadre -8)/pas)
    #print("nbsurligne et collonne :" + str (nbsurligne) + "," + str(nbsurcollone))
    tabres = [[(0,0)]*nbsurcollone for k in range(nbsurligne)]
    for i in range(nbsurligne):
        for j in range(nbsurcollone):
            x1 = x +  i *pas + 4 + 3* encadre
            y1 = y + j *pas + 4 + 3* encadre
            (x2,y2) = trouver_pos_newimage(im1,int(x1) ,int(y1) ,cadre,cadre,im2,int(x1 - encadre) ,int(y1 - encadre),int(cadre + 2*encadre),int(cadre + 2*encadre))
            #teta = np.arctan((x2-x)/(y2-y))
            vx2 = (x1 - x2) / temps     #en pixel par heure
            vy2 = (y1 - y2) / temps
            if (int(x1 - 0.25*pas) <0 or int(y1 - 0.25*pas) <0) :
                    print ("x1 ou y1 sur le bord")

            (x3,y3) = trouver_pos_newimage(im2,int(x2) ,int(y2) ,cadre,cadre,im3,int(x2- encadre) ,int(y2- encadre),int(cadre + 2*encadre),int(cadre + 2*encadre))
            vx3 = (x2 - x3) / temps     #en pixel par heure
            vy3 = (y2 - y3) / temps
            if (int(x2 - 0.25*pas) <0 or int(y2 - 0.25*pas) <0) :
                    print ("x2 ou y2 sur le bord")

            (x4,y4) = trouver_pos_newimage(im3,int(x3) ,int(y3) ,cadre,cadre,im4,int(x3- encadre) ,int(y3- encadre),int(cadre + 2*encadre),int(cadre + 2*encadre))
            vx4 = (x3 - x4) / temps     #en pixel par heure
            vy4 = (y3 - y4) / temps
            if (int(x3 - 0.25*pas) <0 or int(y3 - 0.25*pas) <0) :
                    print ("x3 ou y3 sur le bord")
            tabres[i][j] = ((vx2+vx3+vx4)/3,(vy2+vy3+vy4)/3)

    return tabres

'''
def afficher_venttab(tab,im,x,y,pas):
    for i,a in enumerate(tab):
        for j,(vx,vy,x1,y1,x2,y2,x3,y3,x4,y4,r,e) in enumerate(a):
            posx = pas * (i + 0.25) + x
            posy = pas * (j + 0.25) + y
            norme = int(np.sqrt(vx * vx + vy * vy))*1000
            for loop in range(1, norme):
                try:
                    im.putpixel((int(posx-200 + vx * loop / 3), int(posy-200 + vy * loop / 3)), (230, 30, 30))
                except:
                    "rien"
            try:
                im.putpixel((int(posx-200), int(posy-200)), (255, 255, 255))
            except:
                "rien"
            try:
                im.putpixel((int(x2 - 200), int(y2 - 200)), (255, 0, 255))
            except:
                "rien"
            try:
                im.putpixel((int(x3-200), int(y3-200)), (0, 255, 0))
            except:
                "rien"
            try:
                im.putpixel((int(x4-200), int(y4-200)), (0, 0, 255))
            except:
                "rien"
    im.show()
'''

def tab_vent_simplifié(tab):
    n = len(tab)
    res = [[0]*n for i in range(n)]
    for i,a in enumerate(tab):
        for j,(vx,vy) in enumerate(a):
            res[i][j] = (vx,vy)
    return res

def nbjour (mois, jour) : # nombre de jour après le 2 mai
    nbjourdanslemois = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    resnbjour = 0
    for i in range(5,mois) :
        resnbjour += nbjourdanslemois[i]
    resnbjour += jour - 2
    return resnbjour


def save_dans_fichier_txt(imname, imtype, nbjours, debmois, debjour, debut, fin, coorddeb, taille, nbiterx,nbitery):
    pas = 5
    temps = 15
    timer1 = time.time()
    tab_im = tableau_images(imname, imtype, nbjours, debmois, debjour, debut, fin, coorddeb, taille, nbiterx,nbitery)
    print(len(tab_im[0][0][0]))
    print("temps de chargement des images : " + str(time.time()-timer1))

    timer2 = time.time()
    for i in range(nbjours):
        #nametxtfile = "./Vents/Vents-" + str(debjour) + str(debmois) + "+" + str(i) + "-" + str(coorddeb[0]) + "-" + str(coorddeb[1]) + "-" + str(taille[0]) + "-" + str(pas) + "-" + str(nbiterx) + str(nbitery) + ".txt"
        nametxtfile = "./Vents/Vents-"  + "25+" + str(nbjour(debmois,debjour) + i) + "-" + str(coorddeb[0]) + "-" + str(coorddeb[1]) + "-" + str(taille[0]) + "-" + str(pas) + "-" + str(nbiterx) + str(nbitery) + ".txt"

        f = open(nametxtfile, "w+")
        for n in range(nbcasex):
            for m in range(nbcasey):
                for j in range(len_seq+decalage-4):
                    av = tab_vent_simplifié(grand_tableau(tab_im[i][n][m][j], tab_im[i][n][m][j+1], tab_im[i][n][m][j+2], tab_im[i][n][m][j+3], (0,0), taille, temps, pas))
                    for x in av:
                        for y in x:
                            for z in y:
                                f.write(str(z))
                                f.write('fin_val')
                            f.write('fin_coord')
                        f.write('fin_ligne')
                    f.write('fin_im')
                    print(("case ("+ str(n) +","+str(m)+")"+str(i+1) + '/' + str(nbjours) + ', ' + str(j+1) + '/' + str(len_seq+decalage-4),time.time() - timer2))
                f.write('fin_casex')
            f.write('fin_casey')
        f.write('fin_jour')
        f.close()

    #partie info
    '''
    f.write('fin_info')
    f.write(str(coorddeb[0]))
    f.write('fin_info')
    f.write(str(coorddeb[1]))
    f.write('fin_info')
    f.write(str(nbcasex))
    f.write('fin_info')
    f.write(str(nbcasey))
    f.write('fin_info')
    f.write(str(pas))
    '''


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
                        res[posx][posy].append(np.array(imtemp))
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '44' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '44' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long,
                                          ydeb + (posy + 1) * larg))
                        res[posx][posy].append(np.array(imtemp))
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '42' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '42' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long,
                                          ydeb + (posy + 1) * larg))
                        res[posx][posy].append(np.array(imtemp))
            elif path.exists(imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype):
                im = Image.open(imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype)
                for posx in range(nbiterx):
                    for posy in range(nbitery):
                        imtemp = im.crop((xdeb + posx * long, ydeb + posy * larg, xdeb + (posx + 1) * long,
                                          ydeb + (posy + 1) * larg))
                        res[posx][posy].append(np.array(imtemp))
            else:
                print(
                    '/!\ manque : ' + imname + stringmois + stringjour + rajout + str(a) + str(b) + '43' + '.' + imtype)
            if b >= 57:
                i = (a + 1, 12)
            else:
                i = (a, b + 15)
        tabres.append(res)
    return tabres



#afficher_rogné(im1,200,200,200,200)
#afficher_rogné(im2,200,200,200,200)
#afficher_rogné(im3,200,200,200,200)
#afficher_rogné(im4,200,200,200,200)

#timer = time.time()
#tab = grand_tableau(im1,im2,im3,im4,200,200,200,200,20,20)
#print(time.time()-timer)
#print(tab)

#afficher_venttab(tab,Image.new("RGB", (200,200), "black"),200,200,20)
#print(time.time()-timer)
#print(tab_vent_simplifié(tab))


nbjours = 6  # on peut aller jusqu'à 64 ou 65 (nombre de jours en banque de données à partir du 5 juin)
dimobjectif = (1200, 1400)
dimquimarche = (220, 220)
#dimquimarche = (1170, 1170)
batchsize = 1
len_seq = 28
decalage = 30 - len_seq  # (décalage de temps entre la première image et l'image attendue)
nbcasex = 2
nbcasey = 4
long, larg = dimquimarche


save_dans_fichier_txt("./Images/Total/VIS8_MSG4-SEVI-MSG15-0100-NA-2019", "jpg", nbjours, 5, 14, (9, 42), (17, 28), (700,1700),(long, larg), nbcasex, nbcasey)

'''
"Pour test"
tab_im = tableau_images("./Images/Total/VIS8_MSG4-SEVI-MSG15-0100-NA-2019", "jpg", nbjours, 8, 15, (9, 42), (17, 28), (700,1700),(long, larg), nbcasex, nbcasey)
for i,x in enumerate(tab_im[0][0][0]):
    if i<5:
        Image.fromarray(np.uint8(x)).show()
i = 0
n = 0
m = 0
j = 0
pas = 5
temps = 15
tab_vent = tab_vent_simplifié(grand_tableau(tab_im[i][n][m][j], tab_im[i][n][m][j+1], tab_im[i][n][m][j+2], tab_im[i][n][m][j+3], (0,0), (long, larg), temps, pas))
TAB = tab_vent
print(TAB)
print ("longueur")
print (len(TAB))

X, Y, U, V , C = [], [], [], [], []
for i, a in enumerate(TAB):
    for j, (vx, vy) in enumerate(a):
        #X.append(i * pas)
        #Y.append(- j * pas)
        #U.append(-vy)
        #V.append(vx)
        X.append(j * pas)
        Y.append(-i * pas)
        U.append(-vy)
        V.append(vx)
        #C.append((i % 2 , 0 , j % 2))
        C.append((0,0,0))
plt.figure()
plt.quiver(X, Y, U, V, color = C)
plt.show()
'''