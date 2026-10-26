import nibabel as nib


def load(path):
    return nib.load(path).get_fdata()


# https://niivue.github.io/niivue-demo-images/mni152.nii.gz
# data = load("./mni152.nii/mni152.nii")

# data = load("./icbm_avg_152_t1_tal_lin.nii/icbm_avg_152_t1_tal_lin.nii")
# data = load("./minimal.nii/minimal.nii")
# data = load("./avg152T1_RL_nifti.nii/avg152T1_RL_nifti.nii")
# data = load("./mni_icbm152_t1_tal_nlin_asym_09c.nii/mni_icbm152_t1_tal_nlin_asym_09c.nii")
data = load("./mni_icbm152_t1_tal_nlin_asym_09b_hires.nii/mni_icbm152_t1_tal_nlin_asym_09b_hires.nii")
