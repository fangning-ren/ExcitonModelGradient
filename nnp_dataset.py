import os
import numpy as np
import h5py
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import random_split


class DimerCouplingANIDataset(Dataset):

    periodic_table = """
    H                                                                   He
    Li  Be                                          B   C   N   O   F   Ne
    Na  Mg                                          Al  Si  P   S   Cl  Ar
    K   Ca  Sc  Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y   Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """
    lanthanides = "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu"
    actinides = "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No Lr"

    def __init__(self, data_path, 
                 transform=None, 
                 target_property='1e(1)', 
                 approx_property=None,
                 species_order = None, 
                 reverse_belonging = False,
                 align_coupling_sign = False,
                 use_belonging = False, # deprecated 
                 ):
        self.data_path = data_path

        self.species_order = DimerCouplingANIDataset.periodic_table.split() if species_order is None else species_order
        # self.use_belonging= use_belonging
        self.reverse_belonging = reverse_belonging
        self.target_properties = [target_property, ] if not isinstance(target_property, list) else target_property
        self.approx_properties = [approx_property, ] if not isinstance(approx_property, list) else approx_property
        self.approx_properties = [p for p in self.approx_properties if p is not None]

        if transform == "reverse_phase":
            transform = self._reverse_phase
        elif transform == "random_phase":
            transform = self._random_phase
        elif transform == "absolute_value":
            transform = self._absolute_value
        else:
            transform = None
        self.transform = transform

        self.species = None
        self.belongings = None
        self.coordinates = None
        self.species_with_belonging = None
        self.zeros = 0.0
        self.targets = {}
        self.approxs = {}

        self._load_data()
        if align_coupling_sign:
            self._align_coupling_sign()

    def __getitem__(self, index):
        properties = {
            'species': self.species[index],
            'coordinates': self.coordinates[index],
            "belongings": self.belongings[index],
            "species_with_belonging": self.species_with_belonging[index],
            "zeros": torch.zeros_like(self.belongings[index]).to(torch.float32),
            "targets": None,
            "approxs": None,
        }
        select_targets = []
        if len(self.target_properties) > 0:
            for key in self.target_properties:
                select_targets.append(self.targets[key][index])
            properties["targets"] = select_targets[0].unsqueeze(-1) if len(select_targets) == 1 else torch.stack(select_targets, dim=-1)
        select_approxs = []
        if len(self.approx_properties) > 0:
            for key in self.approx_properties:
                select_approxs.append(self.approxs[key][index])
            properties["approxs"] = select_approxs[0].unsqueeze(-1) if len(select_approxs) == 1 else torch.stack(select_approxs, dim=-1)
        else:
            properties["approxs"] = torch.zeros_like(properties["targets"])
        if self.transform:
            properties = self.transform(properties)
        return properties

    def __len__(self):
        return len(self.species)
    
    def split(self, *portion):
        # remove the possible None from portion
        portion = [p for p in portion if p is not None]
        lengths = [int(len(self) * p) for p in portion]
        remainder = len(self) - sum(lengths)
        if remainder > 0:
            lengths.append(remainder)
        return random_split(self, lengths)
    
    def species_to_indices(self, species_order=None):
        if isinstance(self.species, torch.Tensor):
            return
        if species_order is not None:
            self.species_order = species_order
        indices = []
        species = self.species
        belongings = self.belongings
        for sample in species:
            indices.append([self.species_order.index(s) for s in sample])
        del species
        indices = torch.tensor(indices)
        self.species = indices
        
        self.species_with_belonging = self.species.clone()
        self.species_with_belonging += len(self.species_order) * (belongings - 1)

    def reverse_key(self, skey:str):
        # replace the "1" in skey with "2"
        # replace the "2" in skey with "1"
        # prevent replace the "1" by "2" then replace the "2" by "1"
        # the number in the parenthesis should be kept
        # charge transfer label should be rearranged. e.g. 2+1- -> 1-2+, 2-1+ -> 1+2-
        skey0 = skey
        skey = skey.replace("(1)", "(I)").replace("(2)", "(II)")
        skey = skey.replace("1", "@").replace("2", "1").replace("@", "2")
        skey = skey.replace("(I)", "(1)").replace("(II)", "(2)")
        skey = skey.replace("2+1-", "1-2+").replace("2-1+", "1+2-")
        print(f"reversing the key {skey0} to {skey}")
        return skey

    def _align_coupling_sign(self):
        self.coupling_keys:List[str] 
        atmaprox_keys = [] 
        sumaprox_keys = []
        coupling_keys = []
        for key in self.coupling_keys: 
            if ("apx_" + key) in self.approxs and ("atmapx_" + key) in self.approxs:
                atmaprox_keys.append("atmapx_" + key)
                sumaprox_keys.append("apx_" + key)
                coupling_keys.append(key)
        for cplkey, apxkey, atmkey in zip(coupling_keys, sumaprox_keys, atmaprox_keys):
            sumapx = self.approxs[apxkey]
            atmapx = self.approxs[atmkey]
            cplval = self.targets[cplkey]
            self.approxs[apxkey] = np.abs(sumapx) * np.sign(cplval)
            self.approxs[atmkey] = atmapx * np.sign(sumapx)[:,None] * np.sign(cplval)[:,None]
            print(f"manually aligning the phase of {apxkey} and {atmkey} with {cplkey}")

    def _load_data(self):
        def convert_to_tensor(data):
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            elif isinstance(data, Iterable):
                data = torch.tensor(data)
            if data.dtype == torch.int32:
                data = data.to(torch.int64)
            elif data.dtype == torch.float64:
                data = data.to(torch.float32)
            return data
        with h5py.File(self.data_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                data = f[key][:]
                if key == 'species' or not isinstance(data, np.ndarray) or data.dtype not in [np.float32, np.float64, np.int32, np.int64, int, float]:
                    data = data.astype(str)
                    setattr(self, key, data)
                elif key == "belongings":
                    if self.reverse_belonging:
                        data = data.max() + 1 - data
                    self.belongings = convert_to_tensor(data)
                elif key == "coordinates":
                    self.coordinates = convert_to_tensor(data)
                elif key.find("apx") != -1:
                    if self.reverse_belonging:
                        key = self.reverse_key(key)
                    self.approxs[key] = convert_to_tensor(data)
                else:
                    if self.reverse_belonging:
                        key = self.reverse_key(key)
                    self.targets[key] = convert_to_tensor(data)
        self.zeros = torch.zeros_like(self.belongings).to(torch.float32)


def create_datasets_single(dspaths, target_property, approx_property, species_order, reverse_belonging = False, train_test_split = 0.8):
    trainings, validations = [], []
    for dspath in dspaths:
        dataset = DimerCouplingANIDataset(
            data_path = dspath,
            target_property=target_property,
            approx_property=approx_property,
            species_order = species_order,
            reverse_belonging=reverse_belonging,
        )
        dataset.species_to_indices(species_order)
        training, validation = dataset.split(train_test_split)
        trainings.append(training)
        validations.append(validation)
    training = ConcatDataset(trainings)
    validation = ConcatDataset(validations)
    return training, validation

def create_datasets(dspaths, target_property, approx_property, species_order, align_coupling_sign = False, train_test_split = 0.8):
    datasets = []
    for dspath in dspaths:
        dataset1 = DimerCouplingANIDataset(
            data_path = dspath,
            target_property=target_property,
            approx_property=approx_property,
            species_order = species_order,
            align_coupling_sign=align_coupling_sign,
            reverse_belonging=False,
        )
        dataset1.species_to_indices(species_order)
        datasets.append(dataset1)
        dataset2 = DimerCouplingANIDataset(
            data_path = dspath,
            target_property=target_property,
            approx_property=approx_property,
            species_order = species_order,
            align_coupling_sign=align_coupling_sign,
            reverse_belonging=True,
        )
        dataset2.species_to_indices(species_order)
        datasets.append(dataset2)
    trainings, validations = [], []
    for dataset in datasets:
        training, validation = dataset.split(train_test_split)
        trainings.append(training)
        validations.append(validation)
    training = ConcatDataset(trainings)
    validation = ConcatDataset(validations)
    return training, validation

def sample_dataset(dataset, n_samples, repeat = False):
    if n_samples is None:
        return dataset
    elif n_samples >= len(dataset) or n_samples <= 0:
        return dataset
    elif not repeat:
        return random_split(dataset, [n_samples, len(dataset) - n_samples])[0]
    n_total = len(dataset)
    n_samples = int(n_samples)
    n_repeat = int(np.ceil(n_total / n_samples))
    subset = random_split(dataset, [n_samples, n_total - n_samples])[0]
    return ConcatDataset([subset] * n_repeat)
    