# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage.metrics import structural_similarity

def Evaluate(processedPath,unprocessedPath):
    SSIM_scores = np.array([])
    processedArray = np.array([f for f in os.listdir(processedPath) if f.endswith('_fake_B.png')])
    unprocessedArray = np.array([f for f in os.listdir(unprocessedPath)])
    for i in range(len(processedArray)):
        img1 = cv2.imread(os.path.join(processedPath,processedArray[i]),cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(unprocessedPath,unprocessedArray[i]),cv2.IMREAD_GRAYSCALE)
        if img2.shape != img1.shape :
            img2 = cv2.resize(img2,img1.shape,interpolation=cv2.INTER_AREA)
        (score, diff) = structural_similarity(img1, img2, full=True)
        SSIM_scores = np.append(SSIM_scores,score)
    histoSSIM = plt.hist(SSIM_scores,100,[0.4,1.0])
    plt.show()
Evaluate("C:/Users/conti/Documents/pytorch-CycleGAN-and-pix2pix/results/t1w_cyclegan_naive_correct/test_latest/images","C:/Users/conti/Desktop/Progetto_Pattern/DataSets/DS_A_PIOP1_JPEG")
# %%
