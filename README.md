# Laminar fMRI course

Hands-on exercises for analysing layer-specific (laminar) fMRI data, written for the master's-level course *Advanced Neuroimaging* [course name and university: please check]. The exercises take students from quality control of submillimeter 7T data to depth-resolved connectivity and task effects across cortical layers.

Author: J. Karolis Degutis (University of Geneva and EPFL). Website: [karolisdegutis.com](https://www.karolisdegutis.com)

## What you will learn

Human fMRI at 7 Tesla can resolve signals at different depths of the cortex: superficial, middle and deep. Feedforward and feedback connections enter and leave the cortex at different depths, so laminar fMRI gives a non-invasive window on the direction of communication in the brain. The two exercises cover the main steps of a laminar analysis.

| Notebook | Topic | You will |
|---|---|---|
| [`exercise_01_laminar_fmri_tutorial.ipynb`](exercise_01_laminar_fmri_tutorial.ipynb) | Quality control, layer generation and laminar connectivity | check anatomical–functional alignment; overlay depth maps and Schaefer-400 parcels on the anatomy; generate equi-volume layers from a gray-matter rim with [LAYNII](https://github.com/layerfMRI/LAYNII); extract parcel-wise time courses for three depths; compute layer-specific and full laminar (1200 × 1200) connectivity matrices |
| [`exercise_02_laminar_fmri_tutorial.ipynb`](exercise_02_laminar_fmri_tutorial.ipynb) | Task-based laminar fMRI | analyse working-memory data from [Degutis et al. (2024), *Communications Biology*](https://www.nature.com/articles/s42003-024-06780-8); extract depth-resolved time courses in prefrontal regions of interest; compare high versus low memory load across layers; test layer differences with paired statistics |

Each notebook has a companion module (`laminar_ex1_utils.py`, `laminar_ex1_utils_connectivity.py`, `laminar_ex2_utils.py`) with short, readable functions written for teaching.

## Data

The notebooks expect the course data in the repository folder:

```text
data_Ex1/      # Exercise 1: mean EPI, T1 in EPI space, rim, depth map, Schaefer parcels, layer time courses
data_ex2/      # Exercise 2: subjects S01–S09 with load-dependent responses, ROI masks and layer maps
```

The data are not included in this repository. [Describe where students can download them, e.g. a Zenodo or OpenNeuro link, or "provided on the course platform".]

## Getting started

### Option 1: Docker (recommended)

A ready-made image with all dependencies, including LAYNII, is built automatically from this repository:

```bash
docker pull ghcr.io/degutis/laminar_fmri_course:latest
docker run -p 8888:8888 -v "$PWD":/home/jovyan/work ghcr.io/degutis/laminar_fmri_course:latest
```

The image is based on the [Renku](https://renkulab.io) Python image, so the course can also run on RenkuLab.

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate layerfmri-course
pip install git+https://github.com/MICA-MNI/ENIGMA.git
jupyter lab
```

Then open `exercise_01_laminar_fmri_tutorial.ipynb`.

## Requirements

Python 3.11 with NumPy, SciPy, pandas, Matplotlib, scikit-learn, scikit-image, nibabel, nilearn, statsmodels and LAYNII (all listed in `environment.yml`), plus the [ENIGMA toolbox](https://github.com/MICA-MNI/ENIGMA) for cortical plotting.

## Citation

If you use these materials, please cite:

> Degutis JK. Laminar fMRI course [Computer software]. GitHub; 2026. https://github.com/degutis/laminar_fmri_course

For the Exercise 2 data:

> Degutis JK, et al. Dynamic layer-specific processing in the prefrontal cortex during working memory. *Communications Biology* 7 (2024). https://doi.org/10.1038/s42003-024-06780-8

## License

MIT 2026 
