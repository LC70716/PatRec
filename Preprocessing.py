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


def Nifti2JPG(inputPath, outputPath, denominator=2, size=10):
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
    """
    counter = 0
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
                        nii_data[:, :, slice] = cv2.normalize(
                            nii_data[:, :, slice],
                            None,
                            255,
                            0,
                            cv2.NORM_MINMAX,
                            cv2.CV_8U,
                        )  # each slice is normalized separately
                        cv2.imwrite(
                            outputPath + str(counter) + str(slice) + ".jpg",
                            nii_data[:, :, slice],
                        )
                counter += 1


Nifti2JPG(
    inputPath="C:/Users/conti/Desktop/Progetto_Pattern/DataSets/DS_A_PIOP2",
    outputPath="C:/Users/conti/Desktop/Progetto_Pattern/Train/TrainA",
)

# %%
