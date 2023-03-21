import nibabel as nib
import numpy as np
from dipy.segment.mask import applymask
import pandas as pd
import os
import glob
import csv
demos = pd.read_table('../Diff_analysis/demographics.txt')
healthy = demos[demos['Group']=='healthy']['Subject label'].values
mci = demos[demos['Group']=='MCI']
subs = []
tag = 'all' #Set to MCI for MCI only
with open('../Diff_analysis/subs.txt') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        subs.append(row[0])
subs_2 = []
for sub in subs:
    if not 'session1' in sub:
        if tag == 'mci':
            if mci['Subject label'].str.contains(sub[:-1]).any():
                subs_2.append(sub)
        else:
            subs_2.append(sub)
mask = nib.load('/software/fsl/6.0.4/b1/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz').get_fdata().astype(bool)
affine = nib.load('/software/fsl/6.0.4/b1/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz').affine

def icc(visit_1_arr, visit_2_arr):
    #Define ICC function - taken from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0219854#sec036
    k = 2 #Number of timepoints (raters)
    n = visit_1_arr.shape[3] #Number of subjects
    x_hat = 1/(k*n) * np.sum(visit_1_arr + visit_2_arr, axis = -1) #Grand total mean across measurements and subjects
    S = 1/k * (visit_1_arr + visit_2_arr) #Mean for each subject
    M_1 = 1/n * np.sum(visit_1_arr, axis = -1) #Mean for first timepoint
    M_2 = 1/n * np.sum(visit_2_arr, axis = -1) #Mean for second timepoint
    M = np.append(M_1[...,None], M_2[...,None], axis = -1) #Stack timepoint averages for calculation
    SSBS = k*np.sum((S - x_hat[...,None])**2, axis = -1) #Sum of squares between subjects, multiply times k for k timepoints
    SSBM = n * np.sum((M - x_hat[...,None])**2, axis = -1) #Sum of squares between measurements
    SST = np.sum((visit_1_arr - x_hat[...,None])**2, axis = -1) + np.sum((visit_2_arr - x_hat[...,None])**2, axis = -1) #Total sum of squares
    SSE = SST - SSBS - SSBM #Sum of square of the errors (residuals)
    SSWS = SST - SSBS #Sum of squares within subjects
    MSBS = SSBS/(n - 1)
    MSE = SSE/((n-1)*(k-1))
    MSBM = SSBM/(k-1)
    MSWS = SSWS/(n*(k-1))
    ICC = (MSBS-MSWS)/(MSBS + (k-1)*MSWS)
    return ICC

for accel in ['S1P1','S3P1','S3P2','S6P1','S6P2']:
    i,j=0,0
    print(accel)
    data = nib.load('All_FA_MNI_'+accel+'.nii.gz').get_fdata()
    for a in subs_2:
        ind2 = [i for i, x in enumerate(subs) if x == a]
        ind = [i for i, x in enumerate(subs) if x == a.replace('session2', 'session1')]
        if len (ind2) == 1:
           fa_tmp = data[...,ind]
           if i == 0:
              fa_1 = fa_tmp
           else:
              fa_1 = np.append(fa_1, fa_tmp, axis = -1)
           i+=1
           fa_tmp = data[...,ind2]
           if j == 0:
               fa_2 = fa_tmp
           else:
               fa_2 = np.append(fa_2, fa_tmp, axis = -1)
           j+=1
    fa_icc = icc(fa_1, fa_2)
    fa_icc = applymask(fa_icc, mask)
    if tag == 'mci':
        nib.save(nib.Nifti1Image(fa_icc, affine), 'FA_ICC_mci_MNI_'+accel+'.nii.gz')
    else:
        nib.save(nib.Nifti1Image(fa_icc, affine), 'FA_ICC_all_MNI_'+accel+'.nii.gz')
