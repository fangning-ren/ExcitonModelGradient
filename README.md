## Supplementary material for size-transferable ML exciton model. 

## Authors
Fangning Ren, Xu Chen, Fang Liu

## Dependencies
The code was developed and tested on Python 3.12 with the following packages:
- **python** >= 3.12.0  
- **pytorch** = 2.7.1  
- **tensorboard** >= 2.10.0  
- **torchani** == 2.2  
- **pandas** >= 2.0.3  
- **mdtraj** >= 1.10.0  
- **numba** >= 0.58.0  
- **tqdm** >= 4.66.2  
- **h5py** >= 3.11.0  
- **matplotlib** >= 3.2.2  
- **jupyterlab**  
- **ipykernel**  
- **python-dotenv**  

## Folder structure
Folder structure:
```
├── license.md
├── nnp_dataset.py
├── nnp_models.py
├── nnp_train.py
├── predict_large_aggregates.ipynb
├── predict_oscillator_strength.ipynb
├── README.md
├── environment.yaml
├── data/
│   ├── datasets-ani/
│   │   # training sets in compressed 'hdf5' format for model training
│   │   ├── ...
│   ├── monomer/
│   │   # reference monomer wavefunctions
│   │   ├── ...
│   ├── oos-perylene/
│   │   # out-of-sample test sets for larger perylene aggregates in 'csv' format or mdtraj generated 'hdf5' format
│   │   ├── ...
│   ├── oos-tetracene/
│   │   # out-of-sample test sets for larger tetracene aggregates in 'csv' format or mdtraj generated 'hdf5' format
│   │   ├── ...
├── extract_dimers/
│   # scripts and files for sample the conformers. 
│   ├── get_dimers.py
│   ├── deduplicate.py
│   ├── mmheat.in
│   ├── mmmin.in
│   ├── mmnpt.in
│   ├── perylene-400.prmtop
│   ├── perylene-400.inpcrd
├── models/
|   # trained ML models.
│   ├── perylene-best.pt
│   ├── tetracene-best.pt
├── utils/
|   # utility classes and functions for carrying out the process. 
│   ├── compute_overlap.py
│   ├── compute_overlap_mpi.py
│   ├── molden_convert.py
```

## Usage
Set up the conda environment with the following command. 

```conda env create -f environment.yml```

`nnp_models.py` defines all the model architecture.

`nnp_dataset.py` is the dataset class used in this work.

Run `nnp_train.py` to train the ML model. 

Run the jupyter notebook `predict_large_aggregates.ipynb` to evaluate the model's performance on out-of-sample datasets with larger aggregates. 

Run the jupyter notebook `predict_oscillator_strength.ipynb` to estimate the oscillator strength based on ML-model trained results. 

## Citation
For more details, refer to our [paper](https://pubs.acs.org/doi/10.1021/acs.jpclett.4c03548).