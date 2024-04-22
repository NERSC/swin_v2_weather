# SwinV2_Weather

<img src="assets/nersc.png" width="200px" style="vertical-align: middle;"> <img src="assets/ucberk.png" width="250px" style="vertical-align: middle;"> <img src="assets/indiana.png" width="50px" style="vertical-align: middle;">


This repository contains the code used for "Analyzing and Exploring Training Recipes for Large-Scale Transformer-Based Weather Prediction" \[[paper](https://arxiv.org/abs/XXX.XXXXX)\]

The code was developed by the authors of the preprint: 
[Jared Willard](https://www.linkedin.com/in/jareddwillard/), [Shashank Subramanian](https://www.nersc.gov/about/nersc-staff/data-analytics-services/shashank-subramanian/), [Peter Harrington](https://www.nersc.gov/about/nersc-staff/data-analytics-services/peter-harrington/), [Ankur Mahesh](https://eps.berkeley.edu/people/ankur-mahesh), [Travis O'Brien](https://earth.indiana.edu/directory/faculty/obrien-travis.html), and [William Collins](https://profiles.lbl.gov/11626-william-collins)

SwinV2_Weather is a global data-driven weather forecasting model that provides accurate short to medium-range global predictions at 0.25∘ resolution using a minimally modified SwinV2 transformer. SwinV2_Weather outperforms the forecasting accuracy of the ECMWF Integrated Forecasting System (IFS) deterministic forecast, a state-of-the-art Numerical Weather Prediction (NWP) model, at nearly all lead times for critical large-scale variables like geopotential height at 500 hPa (z500), 2-meter temperature (t2m), and 10-meter wind speed (u10m). 

SwinV2_Weather is based on the original Swin Transformer V2 architecture proposed in Liu et al. \[[2022](http://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html)\], and we adapt the [Hugging Face implementation of the model](https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/models/swin_transformer_v2_cr.py) for this repository.



## Quick Links:

[Model Registry](https://portal.nersc.gov/cfs/dasrepo/swin_model_weights/)

[Precomputed Stats](https://portal.nersc.gov/cfs/dasrepo/swin_stats/)

[ERA5 Dataset Download (NCAR)](https://rda.ucar.edu/datasets/ds633.0/#)

[ERA5 Dataset Download (Nvidia)](https://docs.nvidia.com/deeplearning/modulus/modulus-core/examples/weather/dataset_download/readme.html)

## Training Data:

The model is trained on 73-channels of ERA5 reanalysis data for both single levels \[ [Hersbach 2018](10.24381/cds.adbb2d47) \] and 13 different pressure levels \[ [Hersbach 2018](10.24381/cds.bd0915c6) \]  that is pre-processed and stored into hdf5 files. This data can be aquired from one of the previously mentioned links. 


# Directory Structure for Model Registry
`/swin_model_registry/`

The following subdirectory structure exists for each model used in the preprint. 
```
├── swin_73var_depth12_chweight_invar/
│   ├── global_means.npy (Precomputed stats for normalization)
│   ├── global_stds.npy
│   ├── hyperparams.yaml (model hyperparameter / config information)
│   ├── metadata.json (metadata only used for scoring in Earth2Mip)
│   └── weights.tar (model weights)

```


# Training Configurations

Training configurations can be set up in [config/swin.yaml](swin/AFNO.yaml). The following paths need to be set by the user. These paths should point to the data and stats you downloaded in the steps above:

``` -->
<!-- swin: &backbone
  <<: *FULL_FIELD
  ...
  ...
  orography: !!bool False 
  orography_path: None  # provide path to orography.h5 file if set to true, 
  exp_dir:              # directory path to store training checkpoints and other output
  train_data_path:      # full path to /train/
  valid_data_path:      # full path to /test/
  inf_data_path:        # full path to /out_of_sample. Will not be used while training.
  time_means_path:      # full path to time_means.npy
  global_means_path:    # full path to global_means.npy
  global_stds_path:     # full path to global_stds.npy
  time_diff_means_path: # full path to time_diff_means.npy
  time_diff_stds_path:  # full path to time_diff_stds.npy
  orography_path:       # full path to orography.h5 file
  landmask_path:        # full path to landmask.h5 file

```

# HPC Job Launching

An example launch script for distributed data parallel training on the slurm based HPC cluster perlmutter is provided in ```submit_batch.sh```.

## Inference and Scoring:
For inference and scoring we used [Earth2MIP](https://github.com/NVIDIA/earth2mip). A fork that contains an implementation of our Swin model and directions for inference and scoring can be found at https://github.com/jdwillard19/earth2mip-swin-fork/. 


If you find this work useful, cite it using:
```
@article{TODO BIBTEX CITION HERE ONCE ON ARXIV,
  title={Analyzing and Exploring Training Recipes for Large-Scale Transformer-Based Weather Prediction},
  author={},
  journal={},
  year={2024}
}
```


