# Programme d'identification de chacun des nuages, numérotation, colorisation,
# et recherche du nuage équivalent dans l'image suivante puis calcul du vent en tant que déplacement du centre de gravité

from PIL import Image
import numpy as np
import time

# fonction test
def nuagesfrontière(im):
    (long,larg) = im.size
    for i in range(1,long-1):
        for j in range(1,larg-1):
            if im.getpixel((i,j)) > im.getpixel((i-1,j)) + 7:
                im.putpixel((i,j),255)
            elif im.getpixel((i,j)) > im.getpixel((i+1,j)) + 7:
                im.putpixel((i,j),255)
            elif im.getpixel((i,j)) > im.getpixel((i,j+1)) + 7:
                im.putpixel((i,j),255)
            elif im.getpixel((i,j)) > im.getpixel((i,j-1)) + 7:
                im.putpixel((i,j),255)
    im.show()

# fonction test
def nuagescolorié(im):
    (long, larg) = im.size
    for i in range(1, long - 1):
        for j in range(1, larg - 1):
            if im.getpixel((i, j)) > 57:
                im.putpixel((i, j), 255)
    im.show()

# filtre passe haut => on met à zéro tout les pixels un peu trop sombres
def assombrir(im):
    (long, larg) = im.size
    for i in range(long):
        for j in range(larg):
            if im.getpixel((i, j)) < 30:
                im.putpixel((i, j), 0)


# construit la table des nuages en identifiant les nuages (unifiés)
def tabincertitudes(im,x,y,long,larg):
    im = im.crop((x,y,x+long,y+larg))
    tab = [[0]*larg for i in range(long)]
    indice = 1
    carac = []
    for i in range(long):
        for j in range(larg):
            if tab[i][j] == 0 and im.getpixel((i, j)) != 0:
                (total,densite,gravx,gravy) = unifiernuage(indice,i,j,tab,im,long,larg)
                carac.append((total,densite/total,gravx/densite,gravy/densite))
                indice += 1
    return tab,carac

def unifiernuage(indice,x,y,tab,im,long,larg):
    tab[x][y] = indice
    total = 1
    densite = im.getpixel((x,y))
    gravitex = densite * x
    gravitey = densite * y
    if x+1 < long and im.getpixel((x+1, y)) != 0 and tab[x+1][y] == 0:
        (tot,dens,gravx,gravy) = unifiernuage(indice, x+1, y, tab, im, long, larg)
        total += tot
        densite += dens
        gravitex += gravx
        gravitey += gravy
    if 0 <= x-1 and im.getpixel((x-1, y)) != 0 and tab[x-1][y] == 0:
        (tot,dens,gravx,gravy) = unifiernuage(indice, x-1, y, tab, im, long, larg)
        total += tot
        densite += dens
        gravitex += gravx
        gravitey += gravy
    if y+1 < larg and im.getpixel((x, y+1)) != 0 and tab[x][y+1] == 0:
        (tot,dens,gravx,gravy) = unifiernuage(indice, x, y+1, tab, im, long, larg)
        total += tot
        densite += dens
        gravitex += gravx
        gravitey += gravy
    if 0 <= y-1 and im.getpixel((x, y-1)) != 0 and tab[x][y-1] == 0:
        (tot,dens,gravx,gravy) = unifiernuage(indice, x, y-1, tab, im, long, larg)
        total += tot
        densite += dens
        gravitex += gravx
        gravitey += gravy
    return (total,densite,gravitex,gravitey)


# visualise les nuages en les coloriant
def visualiser(tab):
    (long,larg) = (len(tab),len(tab[0]))
    visu = Image.new("RGB", (long,larg), "black")
    for i in range(long):
        for j in range(larg):
            indice = tab[i][j]
            visu.putpixel((i, j), (246*(indice%7)//7,230*(indice%5)//5,250*(indice%3)//3))
    visu.show()
    return visu

#print(visualiser(tabincertitudes(im1)[0]))


def est_dans(liste,i):
    rep = liste[i]
    for j in range(len(liste)):
        if j != i and liste[j] == rep:
            return (True,j)
    return (False,-1)

# rapproche les nuages d'une liste avec ceux d'une autre liste
def correlation(liste1,liste2):
    passage = [-1]*len(liste1)
    correl = [0]*len(liste1)
    for i,(tot1,dens1,gravx1,gravy1) in enumerate(liste1):
        rmax = 0
        for j,(tot2,dens2,gravx2,gravy2) in enumerate(liste2):
            r = np.exp(-abs((tot2-tot1)/3))*np.exp(-abs(dens2-dens1))*np.exp(-abs(gravx2-gravx1)/2)*(np.exp(-abs(gravx2-gravy1)/2))
            if r > rmax:
                passage[i] = j
                correl[i] = r
                rmax = r
        (bool,num) = est_dans(passage,i)
        if bool:
            if correl[num] > correl[i]:
                passage[i] = -1
                correl[i] = 0
            else:
                passage[num] = -1
                correl[num] = 0
    return [-1] + passage



def modifnumérotab(tab,liste):
    newtab = tab[:]
    (long,larg) = (len(newtab),len(newtab[0]))
    for i in range(long):
        for j in range(larg):
            newtab[i][j] = liste[newtab[i][j]] + 1
    return newtab

# calcule un tableau de vents pour deux listes de nuage
def tableau_vent_sous_nuage(liste1,liste2,passage12,temps):
    tableau_vent = []
    for i,(tot1,dens1,gravx1,gravy1) in enumerate(liste1):
        if passage12[i] != -1:
            (tot2,dens,gravx2,gravy2) = liste2[passage12[i]]
            tableau_vent.append((gravx1,gravy1,(gravx2-gravx1)/temps,(gravy2-gravy1)/temps))
    return tableau_vent

# dessine les vents
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

#nuagescolorié(im)
#nuagesfrontière(im)
#assombrir(im1)
#assombrir(im2)


def main():
    im1 = Image.open('./Images/VIS8_MSG4-SEVI-MSG15-0100-NA-20190615091242.jpg')
    im2 = Image.open('./Images/VIS8_MSG4-SEVI-MSG15-0100-NA-20190615085742.jpg')
    #im3 = Image.open('VIS8_MSG4-SEVI-MSG15-0100-NA-20190615084242.jpg')
    #im4 = Image.open('VIS8_MSG4-SEVI-MSG15-0100-NA-20190615082742.jpg')

    im1.show()

    im1 = im1.crop((450, 350, 550, 400))
    im2 = im2.crop((450, 350, 550, 400))
    #im3 = im3.crop((450, 350, 550, 400))
    #im4 = im4.crop((450, 350, 550, 400))

    im1.show()
    #time.sleep(3)
    im2.show()
    #time.sleep(5)

    assombrir(im1)
    assombrir(im2)
    #assombrir(im3)
    #assombrir(im4)

    tab1 = tabincertitudes(im1,1, 1,100,50)
    tab2 = tabincertitudes(im2,1, 1,100,50)
    #tab1 = tabincertitudes(im1)
    #tab2 = tabincertitudes(im2)
    #tab3 = tabincertitudes(im3)
    #tab4 = tabincertitudes(im4)

    listedelienindices21 = correlation(tab2[1], tab1[1])
    listedelienindices12 = correlation(tab1[1], tab2[1])
    #print(correlation(tab1[1], tab2[1]))
    newtab2 = modifnumérotab(tab2[0], listedelienindices21 )
    visualiser(tab1[0])
    time.sleep(3)
    visualiser(newtab2)
    time.sleep(15)
    tab_vents = tableau_vent_sous_nuage(tab1[1],tab2[1],listedelienindices12,15)
    print(tab_vents)
    #afficher_vents(tab_vents,Image.new("RGB", (100,50), "black"))
    afficher_vents(tab_vents, visualiser(tab1[0]))
    #print(tab3)
    #newtab3 = modifnumérotab(tab3[0], correlation(tab3[1], tab1[1]))
    #newtab4 = modifnumérotab(tab4[0], correlation(tab4[1], tab1[1]))

    #visualiser(newtab3)
    #visualiser(newtab4)

main()