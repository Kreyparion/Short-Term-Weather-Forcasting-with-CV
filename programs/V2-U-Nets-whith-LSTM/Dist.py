from PIL import Image
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse2


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

Temps = "2020-01-27--15-45-29--"
Temps = "2020-01-27--14-43-31--"
Temps = "2020-01-27--13-49-25--"

dist_image('./Images-Res/', Temps, "Attendu-", "Attendu-", "PNG")
dist_image('./Images-Res/', Temps, "x-", "Attendu-", "PNG")
dist_image('./Images-Res/', Temps, "Predict-", "Attendu-", "PNG")
