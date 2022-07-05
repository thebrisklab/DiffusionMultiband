import nibabel as nib
import numpy as np
from dipy.segment.mask import applymask
import pandas as pd
demos = pd.read_table('demographics.txt')
healthy = demos[demos['Group']=='healthy']['Subject label'].values
mci = demos[demos['Group']=='MCI']['Subject label'].values
mask = nib.load('/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz').get_fdata().astype(bool)
affine = nib.load('/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz').affine
def ssa(visit_1_arr, visit_2_arr):
    one = 0.5*np.sum((visit_1_arr + visit_2_arr)**2, axis = 3)
    two = 1/(visit_1_arr.shape[3] * 2) * np.sum(visit_1_arr + visit_2_arr, axis = 3)**2
    return one - two
def sst(visit_1_arr, visit_2_arr):
    tmp = np.append(visit_1_arr, visit_2_arr, axis = 3)
    group_mean = np.mean(tmp, axis = 3)
    val = np.sum((visit_1_arr - group_mean[...,None])**2 + (visit_2_arr - group_mean[...,None])**2, axis = 3)
    return val
def icc(visit_1_arr, visit_2_arr):
    ssa_v = ssa(visit_1_arr, visit_2_arr)
    sst_v = sst(visit_1_arr, visit_2_arr)
    return ssa_v/sst_v

for accel in ['S1P1','S3P1','S3P2','S6P1','S6P2']:
    i,j=0,0
    print(accel)
    for a in mci:
        if os.path.exists(a.replace('*','1')+'/MNI_Reg/dti/md_'+accel+'.nii.gz'):
            if a != 'BIS_s814_session*':
                for visit in [1,2]:
                    if visit==1:
                        fa_tmp = nib.load(a.replace('*','1')+'/MNI_Reg/dti/md_'+accel+'.nii.gz').get_fdata()[...,None]
                        if i == 0:
                            fa_1 = fa_tmp
                        else:
                            fa_1 = np.append(fa_1, fa_tmp, axis = -1)
                        i+=1
                    else:
                        fa_tmp = nib.load(a.replace('*','2')+'/MNI_Reg/dti/md_'+accel+'.nii.gz').get_fdata()[...,None]
                        if j == 0:
                            fa_2 = fa_tmp
                        else:
                            fa_2 = np.append(fa_2, fa_tmp, axis = -1)
                        j+=1
    fa_icc = icc(fa_1, fa_2)
    fa_icc = applymask(fa_icc, mask)
    nib.save(nib.Nifti1Image(fa_icc, affine), 'ICC/MCI/MD_ICC_all_'+accel+'.nii.gz')