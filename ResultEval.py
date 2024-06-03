# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage.metrics import structural_similarity

def reject_outliers(data, m = 3.5):
    abs_deviations = np.abs(data - np.median(data))
    MAD = np.median(abs_deviations)
    score = 0.6745*abs_deviations/MAD 
    return data[score<m]

def Evaluate(processedPath, unprocessedPath,showPlots=True):
    SSIM_scores = np.array([])
    PSNR_scores = np.array([])
    SSIM_scores_NoOut = np.array([])
    PSNR_scores_NoOut = np.array([])
    SNR_ratios = np.array([])
    processedArray = np.array(
        [f for f in os.listdir(processedPath) if f.endswith("_fake_B.png")]
    )
    unprocessedArray = np.array([f for f in os.listdir(unprocessedPath)])
    for i in range(len(processedArray)):
        img1 = cv2.imread(
            os.path.join(processedPath, processedArray[i]), cv2.IMREAD_GRAYSCALE
        )
        img2 = cv2.imread(
            os.path.join(unprocessedPath, unprocessedArray[i]), cv2.IMREAD_GRAYSCALE
        )
        if img2.shape != img1.shape:
            img2 = cv2.resize(img2, img1.shape, interpolation=cv2.INTER_AREA)
        (score, diff) = structural_similarity(img1, img2, full=True)
        old_snr = np.nanmean(np.where(np.isclose(img2,0), np.nan, img2))/np.nanstd(np.where(np.isclose(img2,0), np.nan, img2))
        new_snr = np.nanmean(np.where(np.isclose(img1,0), np.nan, img1))/np.nanstd(np.where(np.isclose(img1,0), np.nan, img1))
        PSNR_scores = np.append(PSNR_scores, cv2.PSNR(img1, img2))
        SSIM_scores = np.append(SSIM_scores, score)
        SNR_ratios = np.append(SNR_ratios,old_snr/new_snr)
    mean_SSIM = np.mean(SSIM_scores)
    std_SSIM = np.std(SSIM_scores)
    mean_PSNR = np.mean(PSNR_scores)
    std_PSNR = np.std(PSNR_scores)
    SSIM_scores_NoOut = reject_outliers(SSIM_scores)
    PSNR_scores_NoOut = reject_outliers(PSNR_scores)
    SNR_ratios_NoOut = reject_outliers(SNR_ratios)
    mean_SSIM_NoOut = np.mean(SSIM_scores_NoOut)
    std_SSIM_NoOut = np.std(SSIM_scores_NoOut)
    mean_PSNR_NoOut = np.mean(PSNR_scores_NoOut)
    std_PSNR_NoOut = np.std(PSNR_scores_NoOut)
    mean_SNR_ratio = np.mean(SNR_ratios)
    std_SNR_ratio = np.std(SNR_ratios)
    mean_SNR_ratio_NoOut = np.mean(SNR_ratios_NoOut)
    std_SNR_ratio_NoOut = np.std(SNR_ratios_NoOut)
    print("mean SSIM = ", mean_SSIM, " +- ", std_SSIM)
    print("mean PSNR = ", mean_PSNR, " +- ", std_PSNR)
    print(" outliers rejected mean SSIM = ", mean_SSIM_NoOut, " +- ", std_SSIM_NoOut)
    print("outliers rejected mean PSNR = ", mean_PSNR_NoOut, " +- ", std_PSNR_NoOut)
    print("mean SNR ratio = ", mean_SNR_ratio, " +- ", std_SNR_ratio)
    print("outliers rejected mean SNR = ", mean_SNR_ratio_NoOut, " +- ", std_SNR_ratio_NoOut)
    if showPlots == True:
        f = plt.figure(1)
        histoSSIM = plt.hist(SSIM_scores, 100, [0.4, 1.0])
        plt.title("SSIM")
        f.show()
        g = plt.figure(2)
        histoPSNR = plt.hist(PSNR_scores,100)
        plt.title("PSNR")
        g.show()
        f1 = plt.figure(3)
        histoSSIM_NoOut = plt.hist(SSIM_scores_NoOut, 100)
        plt.title("SSIM Outliers Rejected")
        f1.show()
        g1 = plt.figure(4)
        histoPSNR_NoOut = plt.hist(PSNR_scores_NoOut,100)
        plt.title("PSNR Outliers Rejected")
        g1.show()
        h = plt.figure(5)
        histoSNR = plt.hist(SNR_ratios, 100)
        plt.title("SNR ratios")
        h.show()
        h1 = plt.figure(6)
        histoSNR_NoOut = plt.hist(SNR_ratios_NoOut, 100)
        plt.title("SNR ratios Outliers rejected")
        h1.show()
        input()
        
                    


Evaluate(
    "C:/Users/conti/Documents/pytorch-CycleGAN-and-pix2pix/results/t1w_cyclegan_naive_correct/test_latest/images",
    "C:/Users/conti/Desktop/Progetto_Pattern/DataSets/DS_A_PIOP1_JPEG",
)
# %%
