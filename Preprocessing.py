# %%
# this is the "image selection block", use this to generate JPGs from nii.gz files
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def PlotFull(nii_data):
    number_of_slices = np.size(nii_data, 2)
    for slice in range(number_of_slices):
        plt.imshow(nii_data[:, :, slice], cmap="gray")
        plt.show()


def PlotAreaOfInterest(nii_data, denominator):
    number_of_slices = np.size(nii_data, 2)
    for slice in range(
        (number_of_slices // denominator) - 10, (number_of_slices // denominator) + 10
    ):
        plt.imshow(nii_data[:, :, slice], cmap="gray")
        plt.show()

def AddNoise(image,std): #THE ACTUAL IMPLEMENTATION MAY NEED TO DIFFER
    noisy = image
    for row in range(np.size(noisy,0)):
        for col in range(np.size(noisy,1)):
            if noisy[row,col] != 0:
               noisy[row,col] += np.random.normal(0.0,std)
    return noisy

def Nifti2JPG(inputPath, outputPath, denominator=2, size=10, noise = False):
    """Generates JPG files given nii.gz files

    Parameters
    ----------
    inputPath : str or os.PathLike
       specification of directory where all the files of interest are shown
    outputPath : str or os.PathLike
       specification of directory where the JPGs will be stored
    denominator : UInt
       slice from where the selection given by range starts (# slice = # slices/denominator)
    size : UInt
       number of slices (the interval is 2*size) before and after the slice pointed by denominator which are to be generated into JPG
    noise : Bool
        if true adds gaussian noise w/ mean and std
    std_weigth : float
        the std deviation is computed as follows : gray scale value * std_weigth
    """
    counter = 0
    frac_snr_tot=np.array([])
    for dirpath, dirnames, filenames in os.walk(inputPath):
        for filename in filenames:
            print(os.path.join(dirpath, filename))
            if filename.endswith(".nii.gz"):
                nii_img = nib.load(os.path.join(dirpath, filename))
                nii_data = nii_img.get_fdata()
                nii_data = np.swapaxes(nii_data, 0, 1)
                nii_data = np.flip(nii_data, 0)
                number_of_slices = np.size(nii_data, 2)
                for slice in range(
                    (number_of_slices // denominator) - size,
                    (number_of_slices // denominator) + size,
                ):
                    if ((number_of_slices // denominator) - size) >= 0 and (
                        (number_of_slices // denominator) + size
                    ) < number_of_slices:
                        if (noise == True):
                            std = np.nanstd(np.where(np.isclose(nii_data[:, :, slice],0), np.nan, nii_data[:, :, slice]))
                            noise_std = std*np.sqrt(2) # 'correct' would be sqrt(3)
                            old_snr_tot = np.nanmean(np.where(np.isclose(nii_data[:, :, slice],0), np.nan, nii_data[:, :, slice]))**2/std**2
                            nii_data[:, :, slice] = AddNoise(nii_data[:, :, slice],noise_std)
                            new_snr_tot = np.nanmean(np.where(np.isclose(nii_data[:, :, slice],0), np.nan, nii_data[:, :, slice]))**2/np.nanstd(np.where(np.isclose(nii_data[:, :, slice],0), np.nan, nii_data[:, :, slice]))**2
                            nii_data[:,:,slice] = np.where(nii_data[:, :, slice] == np.nan, 0, nii_data[:, :, slice])
                            frac_snr_tot=np.append(frac_snr_tot,old_snr_tot/new_snr_tot)
                        min = np.min(nii_data[:,:,slice])
                        max = np.max(nii_data[:,:,slice])
                        nii_data[:,:,slice] = np.uint8(np.floor(((nii_data[:,:,slice] - min)/(max - min))*255))
                        cv2.imwrite(
                            outputPath + str(counter) + str(slice) + ".jpg",
                            nii_data[:, :, slice],
                        )
                counter += 1
                print("mean snr ratio:" + str(np.mean(frac_snr_tot)))


Nifti2JPG(
    inputPath="C:/Users/conti/Desktop/Progetto_Pattern/DataSets/DS_A_PIOP1",
    outputPath="C:/Users/conti/Desktop/Progetto_Pattern/DataSets/DS_A_PIOP1_JPEG/",
    noise=False,
)
  
# %%
