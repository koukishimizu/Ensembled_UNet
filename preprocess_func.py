import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
from bezier_curve import bezier_curve
from tqdm import tqdm
import cv2


modality_name_list = {'t1': '_t1.nii', 
                      't1ce': '_t1ce.nii', 
                      't2': '_t2.nii', 
                      'flair': '_flair.nii'}

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def save_img(slice, label, dir):
    np.savez_compressed(dir, image=slice, label=label)

def norm(slices):
    max = np.max(slices)
    min = np.min(slices)
    slices = 2 * (slices - min) / (max - min) - 1
    return slices

def nonlinear_transformation(slices):

    points_1 = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
    xvals_1, yvals_1 = bezier_curve(points_1, nTimes=100000)
    xvals_1 = np.sort(xvals_1)

    points_2 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_2, yvals_2 = bezier_curve(points_2, nTimes=100000)
    xvals_2 = np.sort(xvals_2)
    yvals_2 = np.sort(yvals_2)

    points_3 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_3, yvals_3 = bezier_curve(points_3, nTimes=100000)
    xvals_3 = np.sort(xvals_3)

    points_4 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_4, yvals_4 = bezier_curve(points_4, nTimes=100000)
    xvals_4 = np.sort(xvals_4)
    yvals_4 = np.sort(yvals_4)

    points_5 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_5, yvals_5 = bezier_curve(points_5, nTimes=100000)
    xvals_5 = np.sort(xvals_5)

    """
    slices, nonlinear_slices_2, nonlinear_slices_4 are source-similar images
    nonlinear_slices_1, nonlinear_slices_3, nonlinear_slices_5 are source-dissimilar images
    """
    nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
    nonlinear_slices_1[nonlinear_slices_1 == 1] = -1
    
    nonlinear_slices_2 = np.interp(slices, xvals_2, yvals_2)

    nonlinear_slices_3 = np.interp(slices, xvals_3, yvals_3)
    nonlinear_slices_3[nonlinear_slices_3 == 1] = -1

    nonlinear_slices_4 = np.interp(slices, xvals_4, yvals_4)

    nonlinear_slices_5 = np.interp(slices, xvals_5, yvals_5)
    nonlinear_slices_5[nonlinear_slices_5 == 1] = -1

    return slices, nonlinear_slices_1, nonlinear_slices_2, \
           nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5


def main(data_root, modality, target_root):
    list_dir = os.listdir(data_root)
    tbar = tqdm(list_dir, ncols=70)
    count = 0

    for name in tbar:
        nib_img = nib.load(os.path.join(data_root, name, name + modality_name_list[modality]))
        nib_mask = nib.load(os.path.join(data_root, name, name + '_seg.nii'))
        
        affine = nib_img.affine.copy()
        
        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        masks[masks != 0] = 1

        w, h, c = 120, 120, slices.shape[2]
        slices_dtype = slices.dtype
        masks_dtype = masks.dtype
        resize_slices = np.zeros((w, h, c), dtype=slices_dtype)
        resize_masks = np.zeros((w, h, c), dtype=masks_dtype)

        for i in range(slices.shape[2]):
            resize_slices[:, :, i] = cv2.resize(slices[:, :, i], (w, h))
            resize_masks[:, :, i] = cv2.resize(masks[:, :, i], (w, h))

        slices = norm(resize_slices)
        slices, nonlinear_slices_1, nonlinear_slices_2, \
        nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5 = nonlinear_transformation(slices)

        if not os.path.exists(os.path.join(target_root, modality + '_ss')):
            os.makedirs(os.path.join(target_root, modality + '_ss'))
        if not os.path.exists(os.path.join(target_root, modality + '_sd')):
            os.makedirs(os.path.join(target_root, modality + '_sd'))

        for i in range(slices.shape[2]):
            """
            Source-Similar
            """
            save_img(slices[:, :, i], resize_masks[:, :, i], os.path.join(target_root, modality + '_ss', 'sample{}_0.npz'.format(count)))
            save_img(nonlinear_slices_2[:, :, i], resize_masks[:, :, i], os.path.join(target_root, modality + '_ss', 'sample{}_1.npz'.format(count)))
            save_img(nonlinear_slices_4[:, :, i], resize_masks[:, :, i], os.path.join(target_root, modality + '_ss', 'sample{}_2.npz'.format(count)))
            """
            Source-Dissimilar
            """
            save_img(nonlinear_slices_1[:, :, i], resize_masks[:, :, i], os.path.join(target_root, modality + '_sd', 'sample{}_0.npz'.format(count)))
            save_img(nonlinear_slices_3[:, :, i], resize_masks[:, :, i], os.path.join(target_root, modality + '_sd', 'sample{}_1.npz'.format(count)))
            save_img(nonlinear_slices_5[:, :, i], resize_masks[:, :, i], os.path.join(target_root, modality + '_sd', 'sample{}_2.npz'.format(count)))
            count += 1

def make_test(data_root, modality, target_root):
    print(f"modality: {modality}")
    
    list_dir = os.listdir(data_root)
    tbar = tqdm(list_dir, ncols=70)
    count = 0
    for name in tbar:
        nib_img = nib.load(os.path.join(data_root, name, name + modality_name_list[modality]))
        nib_mask = nib.load(os.path.join(data_root, name, name + '_seg.nii'))
        
        affine = nib_img.affine.copy()
        
        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        masks[masks != 0] = 1

        w, h, c = 120, 120, slices.shape[2]
        slices_dtype = slices.dtype
        masks_dtype = masks.dtype
        resize_slices = np.zeros((w, h, c), dtype=slices_dtype)
        resize_masks = np.zeros((w, h, c), dtype=masks_dtype)

        for i in range(slices.shape[2]):
            resize_slices[:, :, i] = cv2.resize(slices[:, :, i], (w, h))
            resize_masks[:, :, i] = cv2.resize(masks[:, :, i], (w, h))

        slices = norm(resize_slices)
        # slices, nonlinear_slices_1, nonlinear_slices_2, \
        # nonlinear_slices_3, nonlinear_slices_4, nonlinear_slices_5 = nonlinear_transformation(slices)

        os.makedirs(os.path.join(target_root, modality), exist_ok=True)

        for i in range(slices.shape[2]):
            save_img(slices[:, :, i], resize_masks[:, :, i], os.path.join(target_root, modality, 'sample{}.npz'.format(count)))
            count += 1

if __name__ == '__main__':
    data_root = '/works/data/BRATS-2018/MICCAI_BraTS_2018_Data_Training/train'
    target_root = '/works/data/BRATS-2018/MICCAI_BraTS_2018_Data_Training/npz_data_120_t1ce/train'
    modality = 't1ce'
    main(data_root, modality, target_root)
    del modality_name_list[modality]

    data_root = '/works/data/BRATS-2018/MICCAI_BraTS_2018_Data_Training/test'
    target_root = '/works/data/BRATS-2018/MICCAI_BraTS_2018_Data_Training/npz_data_120_t1ce/test'
    for modality in modality_name_list.keys():
        make_test(data_root, modality, target_root)