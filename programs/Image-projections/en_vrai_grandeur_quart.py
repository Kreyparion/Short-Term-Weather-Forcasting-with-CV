from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt


def deformer(im,quart,pariterogné):     #avec partierognée ((xd,yd),(xf,yf))
    ((xld, yld), (xlf, ylf)) = pariterogné

    nbpixelmax = 1557  #dépend de l'angle limite voulu
    (long,larg) = im.size
    distancepixel = 3  #en km
    tot = (long/2)/1.0116

    longtab = 1860*2
    largtab = 1860*2
    newim = Image.new("L", (longtab//2+1, largtab//2+1), "black")

    tab = [[] for i in range(longtab//2+1)]
    for ii in range(longtab//2+1):
        tab[ii] = [[] for i in range(largtab//2+1)]
    if quart == 1:
        coeflong = 1
        coeflarg = 1
    elif quart == 2:
        coeflong = 0
        coeflarg = 1
    elif quart == 3:
        coeflong = 1
        coeflarg = 0
    elif quart == 4:
        coeflong = 0
        coeflarg = 0

    for i in range(max(long//2 - nbpixelmax*(quart%2),int(xld)),min(long//2 + nbpixelmax*((quart-1)%2),int(xlf))):
        if i % 50 == 0:
            print(i)
        for j in range(max(larg//2 - nbpixelmax*(quart==1 or quart==2),int(yld)),min(larg//2 + nbpixelmax*(quart==3 or quart==4),int(ylf))):

            anglexd = pixel_to_angle((i - long//2)/tot)
            angleyd = pixel_to_angle((j - larg//2)/tot)
            angle = np.arccos(np.cos(anglexd)*np.cos(angleyd))
            #print(anglexd,angleyd,angle)
            if abs(angle) < 2*np.pi*(50/360):
                color = im.getpixel((i,j))
                if color > 7:
                    anglexf = pixel_to_angle((i - long // 2 + 1)/tot)
                    angleyf = pixel_to_angle((j - larg // 2 + 1)/tot)
                    xd = rayonterre * anglexd
                    yd = rayonterre * angleyd
                    xf = rayonterre * anglexf
                    yf = rayonterre * angleyf
                    if xd < 0:
                        xf,xd=xd,xf
                    if yd < 0:
                        yf,yd = yd,yf
                    numpixxd, rxd = divmod(xd, 3)
                    numpixyd, ryd = divmod(yd, 3)
                    numpixxf, rxf = divmod(xf, 3)
                    numpixyf, ryf = divmod(yf, 3)
                    numpixxd = int(numpixxd)
                    numpixxf = int(numpixxf)
                    numpixyd = int(numpixyd)
                    numpixyf = int(numpixyf)
                    tab[int(numpixxd) + longtab//2*coeflong][int(numpixyd) + largtab//2*coeflarg].append((color, airangle(xd, yd, rxd, ryd, xf, yf))) #les quatres angles
                    tab[int(numpixxf) + longtab//2*coeflong][int(numpixyd) + largtab//2*coeflarg].append((color, airangle(xf, yd, rxf, ryd, xd, yf)))
                    tab[int(numpixxd) + longtab//2*coeflong][int(numpixyf) + largtab//2*coeflarg].append((color, airangle(xd, yf, rxd, ryf, xf, yd)))
                    tab[int(numpixxf) + longtab//2*coeflong][int(numpixyf) + largtab//2*coeflarg].append((color, airangle(xf, yf, rxf, ryf, xd, yd)))

                    for loop in range(abs(numpixxd)+1,abs(numpixxf)):
                        tab[np.sign(numpixxd)*loop + longtab//2*coeflong][numpixyd + largtab//2*coeflarg].append((color, airdroit(yd,yf,rxd)))
                    for loop in range(abs(numpixxd)+1,abs(numpixxf)):
                        tab[np.sign(numpixxd)*loop + longtab//2*coeflong][numpixyf + largtab//2*coeflarg].append((color, airdroit(yf,yd,rxf)))
                    for loop in range(abs(numpixyd)+1,abs(numpixyf)):
                        tab[numpixxd + longtab//2*coeflong][np.sign(numpixyd)*loop + largtab//2*coeflarg].append((color, airdroit(xd,xf,ryd)))
                    for loop in range(abs(numpixyd)+1,abs(numpixyf)):
                        tab[numpixxf + longtab//2*coeflong][np.sign(numpixyd)*loop + largtab//2*coeflarg].append((color, airdroit(xf,xd,ryf)))

                    for loop1 in range(abs(numpixxd)+1,abs(numpixxf)):
                        for loop2 in range(abs(numpixyd)+1,abs(numpixyf)):
                            tab[np.sign(numpixxd)*loop1 + longtab//2*coeflong][np.sign(numpixyd) * loop2 + largtab//2*coeflarg].append((color, 9))
    #print(tab)
    for i,tranche in enumerate(tab):
        for j,pix in enumerate(tranche):
            totaire = 0
            res = 0
            for (couleur,aire) in pix:
                res += couleur*aire
                totaire += aire
            if totaire != 0:
                newim.putpixel((i,j),int(res/totaire))

    return newim
    #newim.show()
    #newim.save(imname+"+"+str(quart)+"."+imtype)





def airdroit(ad,af,r):
    if af>ad:
        return r
    else:
        return 3-r


def airangle(xd,yd,rxd,ryd,xf,yf):
    if xf > xd:
        if yf > yd:
            return (3 - rxd) * (3 - ryd)
        else:
            return (3 - rxd) * (ryd)
    else:
        if yf > yd:
            return (rxd) * (3 - ryd)
        else:
            return (rxd) * (ryd)


def pixel_to_angle(x):  #On rentre un pourcentage du rayon terrestre
    Rt = rayonterre
    H = orbite
    return np.arcsin((H*H*x-x*Rt*np.sqrt(x*x*(Rt*Rt-H*H)+H*H))/(x*x*Rt*Rt+H*H))


orbite = 42157
rayonterre = 6371

def tests(im):  #paramètres de calibrage de l'image résultat

    """
    print(pixel_to_angle(1.0116))   #Pas 1 car on prend l'applatissement de la terre (due à sa rotation)
    print(np.arccos(rayonterre/orbite))
    print(1/(np.sin(1.419)))
    print(np.pi/2)
    """

    (long, larg) = im.size
    tot = (long/2)/1.0116   #On prend 1.0116 au lieu de 1 car on prend en compte l'applatissement de la terre (due à sa rotation)

    anglelim1 = pixel_to_angle(0.93681)      #jusqu'a 60 degré, après c'est flou (trouvé par tatonement)
    portionlim1 = 0.93681
    print(portionlim1 * (tot))   # 1718 pixel
    taillelim1 = rayonterre*anglelim1
    nbpixellim1 = taillelim1/3        # Ici on prends comme référence 2250 on divise par la taille d'un pixel 3km
    print(nbpixellim1)
    print(anglelim1*360/(2*np.pi))


    anglelim2 = pixel_to_angle(0.8484658)      #jusqu'a 50 degré, après c'est flou (trouvé par tatonement)
    portionlim2 = 0.8484658
    print(portionlim2 * (tot))   # 1557 pixel
    taillelim2= rayonterre*anglelim2
    nbpixellim2 = taillelim2/3        # Ici on prends comme référence 1860 on divise par la taille d'un pixel 3km
    print(nbpixellim2)
    print(anglelim2*360/(2*np.pi))


def main():
    quart = 1
    imname = "VIS8_MSG4-SEVI-MSG15-0100-NA-20190502115744"
    imtype = "jpg"

    im = Image.open("./Images/Total/VIS8_MSG4-SEVI-MSG15-0100-NA-20190502115744.jpg")
    print(im.size)

    tests(im)
    im = im.crop((53,53,3659,3659))
    timer = time.time()
    newim = deformer(im,quart,((0,0),(4000,4000)))
    print(time.time()-timer)
    newim.save("./Images-Res/" + imname + "." + str(quart) + '.PNG', format='PNG')
main()