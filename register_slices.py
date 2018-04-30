import SimpleITK as sitk
import os


data_dir = '/home/users/adoyle/receptors/'
histological_slice = "QH#HG#MR1s4#L#AGNissl#32#11.jpg"

if __name__ == '__main__':

    fixed_image = sitk.ReadImage(data_dir + histological_slice)

    for filename in os.listdir(data_dir):
        if not filename == histological_slice:
            moving_image = sitk.ReadImage(data_dir + filename)

            result = sitk.Elastix(fixed_image, moving_image, "affine")

            sitk.WriteImage(result, data_dir + moving_image[:-4] + '_registered.png')


