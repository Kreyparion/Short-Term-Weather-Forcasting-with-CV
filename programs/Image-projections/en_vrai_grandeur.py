from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

def test(im):
    (long, larg) = im.size
    for i in range(long):
        for j in range(larg):
            im.getpixel((i,j))

def deformer(im,pariterogné):
    ((xld, yld), (xlf, ylf)) = pariterogné
    rayonterre = 6371
    nbpixelmax = 1718  #dépend de l'image (on travaille sur les mêmes images donc fixée à 1718)
    ((xld, yld), (xlf, ylf)) = pariterogné
    (long,larg) = im.size
    distancepixel = 3  #en km
    tot = (long/2)/1.0116
    longtab = 2224*2
    largtab = 2224*2
    tab = [[] for i in range(longtab)]
    for ii in range(longtab):
        tab[ii] = [[] for i in range(largtab)]
    timer = time.time()

    nxld = int((fctsecrete((xld - long // 2 + 1)/tot) *rayonterre)/3)
    nxlf = int((fctsecrete((xlf - long // 2 + 1)/tot) *rayonterre)/3)
    nyld = int((fctsecrete((yld - long // 2 + 1) / tot) * rayonterre)/ 3)
    nylf = int((fctsecrete((ylf - long // 2 + 1) / tot) * rayonterre)/ 3)

    for i in range(max(long//2 - nbpixelmax,int(nxld+2224)),min(long//2 + nbpixelmax,int(nxlf+2224))):
        if i % 50 == 0:
            print(time.time()-timer,i)
        for j in range(max(larg//2 - nbpixelmax,int(nyld+2224)),min(larg//2 + nbpixelmax,int(nylf+2224))):
            anglexd = fctsecrete((i - long//2)/tot)
            angleyd = fctsecrete((j - larg//2)/tot)
            angle = np.arccos(np.cos(anglexd)*np.cos(angleyd))
            #print(anglexd,angleyd,angle)
            if abs(angle) < 2*np.pi*(60/360):
                color = im.getpixel((i,j))
                anglexf = fctsecrete((i - long // 2 + 1)/tot)
                angleyf = fctsecrete((j - larg // 2 + 1)/tot)
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
                tab[int(numpixxd) + longtab//2][int(numpixyd) + largtab//2].append((color, airangle(xd, yd, rxd, ryd, xf, yf))) #les quatres angles
                tab[int(numpixxf) + longtab//2][int(numpixyd) + largtab//2].append((color, airangle(xf, yd, rxf, ryd, xd, yf)))
                tab[int(numpixxd) + longtab//2][int(numpixyf) + largtab//2].append((color, airangle(xd, yf, rxd, ryf, xf, yd)))
                tab[int(numpixxf) + longtab//2][int(numpixyf) + largtab//2].append((color, airangle(xf, yf, rxf, ryf, xd, yd)))

                for loop in range(abs(numpixxd)+1,abs(numpixxf)):
                    tab[np.sign(numpixxd)*loop + longtab//2][numpixyd + largtab//2].append((color, airdroit(yd,yf,rxd)))
                for loop in range(abs(numpixxd)+1,abs(numpixxf)):
                    tab[np.sign(numpixxd)*loop + longtab//2][numpixyf + largtab//2].append((color, airdroit(yf,yd,rxf)))
                for loop in range(abs(numpixyd)+1,abs(numpixyf)):
                    tab[numpixxd + longtab//2][np.sign(numpixyd)*loop + largtab//2].append((color, airdroit(xd,xf,ryd)))
                for loop in range(abs(numpixyd)+1,abs(numpixyf)):
                    tab[numpixxf + longtab//2][np.sign(numpixyd)*loop + largtab//2].append((color, airdroit(xf,xd,ryf)))

                for loop1 in range(abs(numpixxd)+1,abs(numpixxf)):
                    for loop2 in range(abs(numpixyd)+1,abs(numpixyf)):
                        tab[np.sign(numpixxd)*loop1 + longtab//2][np.sign(numpixyd) * loop2 + largtab//2].append((color, 9))
    #print(tab)

    newim = Image.new("L", (2224*2,2224*2), "black")
    for i in range(long):
        for j in range(larg):
            totaire = 0
            res = 0
            for (couleur,aire) in tab[i][j]:
                res += couleur*aire
                totaire += aire
            if totaire != 0:
                newim.putpixel((i,j),int(res/totaire))
    print(nylf,ylf,fctsecrete((ylf - long // 2 + 1) / tot))
    newim.putpixel((nxld+2224, nyld+2224), 255)
    newim.putpixel((nxlf+2224, nylf+2224), 255)
    return newim





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


def fctsecrete(x):
    Rt = 6371
    H = 42157
    return np.arcsin((H*H*x-x*Rt*np.sqrt(x*x*(Rt*Rt-H*H)+H*H))/(x*x*Rt*Rt+H*H))

