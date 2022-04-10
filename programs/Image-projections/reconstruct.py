from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

def reconstruct(imname,imtype):
    im1 = Image.open(imname + "+1" + "." + imtype)
    im2 = Image.open(imname + "+2" + "." + imtype)
    im3 = Image.open(imname + "+3" + "." + imtype)
    im4 = Image.open(imname + "+4" + "." + imtype)
    #need a fast fonction that can put these images "bout à bout"