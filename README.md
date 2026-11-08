# MNI152

## Dataset

| Dataset | Description | Link |
|---------|----------------------|------|
| Linear ICBM Average Brain (ICBM152) | An average of 152 T1-weighted MRI scans, linearly transformed to Talairach space | [nist.mni.mcgill.ca/icbm-152lin](https://nist.mni.mcgill.ca/icbm-152lin/) |
| mni152.nii.gz | Convenient modern redistribution of the MNI152 template | [niivue demo](https://niivue.github.io/niivue-demo-images/mni152.nii.gz) |
| minimal.nii | A simple ramp image, increasing from 0-63 along the Y axis... provided as an example and test dataset | [NIfTI test data](https://nifti.nimh.nih.gov/nifti-1/data/minimal.nii.gz/file_view.html) |
| avg152T1_RL_nifti.nii | Classic avg152T1 stored in neurological (RAS) convention | [NIfTI test data](https://nifti.nimh.nih.gov/nifti-1/data/avg152T1_RL_nifti.nii.gz/file_view.html) |
| ICBM 2009c Nonlinear Asymmetric | 1x1x1mm template | [NIFTI 57MB](https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09c_nifti.zip) |
| ICBM 2009b Nonlinear Asymmetric** | 0.5x0.5x0.5mm template | [NIFTI 358MB](https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip) |

**Not committed due to filesize

## mni152.py

View brain slices of the MNI152 dataset with visualization of brain slice planes.

```
python mni152.py
```

opens the uncommented `PATH` near the top of `mni152.py`. That default is the 0.5 mm ICBM 2009b nonlinear asymmetric template.

```
python mni152.py path\to\scan.nii
```

opens that file instead. The argument can be any `.nii` or `.nii.gz` path, including a volume that is not in this folder. A relative path is taken from the current directory. `python mni152.py -h` prints the same usage.

The menu between the 3D view and the sliders still switches among the NIfTI files next to the script. A file opened from the command line is included in that list when it lives somewhere else.

![](mni_icbm152_t1_tal_nlin_asym_09b_hires.nii.png)

## mni152-asymmetry.ipynb

Left–right asymmetry of the ICBM 2009c asymmetric T1. A raw mirror subtraction, the same map after a blur (right frontal and left occipital petalia), and a left-versus-right look at the lateral fissure.