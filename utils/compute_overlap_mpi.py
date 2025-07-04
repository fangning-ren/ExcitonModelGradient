import os
import sys
import tqdm
import time
from multiprocessing import Pool, Queue, Process, Manager
from typing import List

import numpy as np
import pandas as pd
import tqdm

try:
    import mdtraj as md
except ImportError:
    print("mdtraj is not installed. Cannot input hdf5 file.")
    md = None

from .molden_convert import *
from .compute_overlap import compute_orbital_overlap_kernal, compute_atomwise_overlap_kernal

def get_transformation_matrix(coord1, coord2):
    # Ensure that the input coordinates are NumPy arrays
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)

    # Calculate the centroids (center of mass) for both coordinate sets
    centroid1 = np.mean(coord1, axis=0)
    centroid2 = np.mean(coord2, axis=0)

    # Translate the coordinates so that their centroids coincide
    translated_coord1 = coord1 - centroid1
    translated_coord2 = coord2 - centroid2

    # Calculate the covariance matrix
    cov_matrix = np.dot(translated_coord2.T, translated_coord1)

    # Perform singular value decomposition (SVD) on the covariance matrix
    U, S, Vt = np.linalg.svd(cov_matrix)

    # Calculate the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure the determinant of the rotation matrix is +1 (to prevent reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Calculate the scaling factor
    scale = np.trace(np.dot(translated_coord2.T, translated_coord2)) / np.trace(np.dot(translated_coord1.T, translated_coord1))
    scale = np.sqrt(scale)

    # Create the transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = scale * R
    transformation_matrix[:3, 3] = centroid2 - scale * np.dot(R, centroid1)

    return scale * R

def read_xyz(fname:str):
    with open(fname) as f:
        n = int(f.readline())
        f.readline()
        coords = np.zeros((n, 3))
        elems  = np.zeros(n, dtype = str)
        for i in range(n):
            line = f.readline().split()
            coords[i] = line[1:]
            elems[i] = line[0]
    return elems, coords

def get_porb_pop(wf:MoldenWavefunction, target_orbital_index:int):
    c, a, p, l, aidx = wf.get_raveled_gtf()
    gtf_info_df = pd.DataFrame(
        columns = ["x", "y", "z", "c", "a", "lx", "ly", "lz", "aidx"],
        index = range(len(c))
    )
    gtf_info_df["x"] = p[:, 0]
    gtf_info_df["y"] = p[:, 1]
    gtf_info_df["z"] = p[:, 2]
    gtf_info_df["c"] = c
    gtf_info_df["a"] = a
    gtf_info_df["lx"] = l[:, 0]
    gtf_info_df["ly"] = l[:, 1]
    gtf_info_df["lz"] = l[:, 2]
    gtf_info_df["aidx"] = aidx

    gtf_coef_df = pd.DataFrame(
        columns = range(wf.C.shape[1]),
        index = range(len(c)),
        data = wf.get_raveled_C()
    )
    porb_index = np.where(gtf_info_df["lx"] + gtf_info_df["ly"] + gtf_info_df["lz"] == 1)[0]
    pgtf_info_df = gtf_info_df.iloc[porb_index].copy()
    pgtf_coef_df = gtf_coef_df.iloc[porb_index].copy()
    pgtf_info_df.index = range(pgtf_info_df.shape[0])
    pgtf_coef_df.index = range(pgtf_coef_df.shape[0])

    target_orb = target_orbital_index
    target_c = pgtf_coef_df.loc[:,target_orb]
    pgtf_info_df.loc[:,"coef"] = target_c
    pgtf_info_df.loc[:,"ptyp"] = np.array(["x" for i in range(pgtf_info_df.shape[0])])
    pgtf_info_df.loc[np.nonzero(pgtf_info_df["lx"].values==1)[0],"ptyp"] = "px"
    pgtf_info_df.loc[np.nonzero(pgtf_info_df["ly"].values==1)[0],"ptyp"] = "py"
    pgtf_info_df.loc[np.nonzero(pgtf_info_df["lz"].values==1)[0],"ptyp"] = "pz"
    pgtf_info_df.loc[:,"indx"] = np.arange(0, pgtf_info_df.shape[0])


    pgtf_first_dedup = pgtf_info_df.loc[:,("x", "y", "z", "c", "a", "aidx")].drop_duplicates()
    pgtf_first_dedup['unique_index'] = range(len(pgtf_first_dedup))
    pgtf_info_df = pd.merge(pgtf_info_df, pgtf_first_dedup, on=("x", "y", "z", "c", "a", "aidx"), how='left')
    pxyzpop = pgtf_info_df.pivot(columns="ptyp", index = "unique_index", values="coef")
    pother = pgtf_first_dedup.set_index("unique_index")
    porb_pop = pd.concat([pother, pxyzpop], axis = 1)
    return porb_pop

def convert_molden_to_porb_pop(wffiles):
    names = sorted(list(set([f.split("_frag_")[0] for f in wffiles])))
    ref_wf = MoldenWavefunction(f"{names[0]}_frag_1.molden")
    ref_hmmo_ppop = get_porb_pop(ref_wf, ref_wf.hmmo)
    multiindex = pd.MultiIndex.from_product([pd.RangeIndex(len(names)), ["h1", "h2", "l1", "l2"], pd.RangeIndex(ref_hmmo_ppop.index.shape[0])])
    df = pd.DataFrame(index=multiindex, columns=[ref_hmmo_ppop.columns])

    for i, name in enumerate(tqdm.tqdm(names)): 
        mn1_wf_path = f"{name}_frag_1.molden" 
        mn2_wf_path = f"{name}_frag_2.molden"
        mn1_wf = MoldenWavefunction(mn1_wf_path)
        mn2_wf = MoldenWavefunction(mn2_wf_path)

        mn1_hmmo_ppop = get_porb_pop(mn1_wf, mn1_wf.hmmo)
        mn2_hmmo_ppop = get_porb_pop(mn2_wf, mn2_wf.hmmo)
        mn1_lumo_ppop = get_porb_pop(mn1_wf, mn1_wf.lumo)
        mn2_lumo_ppop = get_porb_pop(mn2_wf, mn2_wf.lumo)
        
        for label, ppop in zip(["h1", "h2", "l1", "l2"], [mn1_hmmo_ppop, mn2_hmmo_ppop, mn1_lumo_ppop, mn2_lumo_ppop]):
            df.loc[(i, label, slice(None))] = ppop.values
    df["aidx"] = df["aidx"].astype(int)
    return df

def compute_overlap_single_process(
        xyzs:np.ndarray,
        df_hmmopop:pd.DataFrame,
        df_lumopop:pd.DataFrame,
        refcoord:np.ndarray,
        use_tqdm:bool=False
        ):
    mnatom = refcoord.shape[0]
    refcenter = np.mean(refcoord, axis = 0)

    hmmopipole = np.array(df_hmmopop.loc[:,("px", "py", "pz")])
    lumopipole = np.array(df_lumopop.loc[:,("px", "py", "pz")])
    gtfcoord = np.array(df_hmmopop.loc[:,("x", "y", "z")]) * 0.529177
    gtfcoef0 = np.array(df_hmmopop.loc[:,"c"])
    gtfcontr = np.array(df_hmmopop.loc[:,"a"])
    gtfatidx = np.array(df_hmmopop.loc[:,"aidx"])

    gtfcoord = np.repeat(gtfcoord, 3, axis=0)
    gtfcoef0 = np.repeat(gtfcoef0, 3)
    gtfcontr = np.repeat(gtfcontr, 3)
    gtfatidx = np.repeat(gtfatidx, 3)

    gtflxyz = np.tile([[1,0,0],[0,1,0],[0,0,1]], reps = (df_hmmopop.shape[0], 1))

    atomwise_overlaps_hmmo1 = []
    atomwise_overlaps_hmmo2 = []
    atomwise_overlaps_lumo1 = []
    atomwise_overlaps_lumo2 = []

    iteratorr = tqdm.tqdm(range(len(xyzs))) if use_tqdm else range(len(xyzs))
    for i in iteratorr:
        coords = xyzs[i]
        coord1, coord2 = coords[:mnatom], coords[mnatom:]
        center1, center2 = np.mean(coord1, axis = 0), np.mean(coord2, axis = 0)
        R1 = get_transformation_matrix(coord1, refcoord)
        R2 = get_transformation_matrix(coord2, refcoord)
        # gtfcenter1 = ((gtfcoord - refcenter) @ R1.T + center1) / 0.529177
        # gtfcenter2 = ((gtfcoord - refcenter) @ R2.T + center2) / 0.529177
        gtfcenter1 = coord1[gtfatidx - 1] / 0.529177
        gtfcenter2 = coord2[gtfatidx - 1] / 0.529177

        atomwise_overlap_hmmo1 = np.zeros(coord1.shape[0])
        atomwise_overlap_hmmo2 = np.zeros(coord2.shape[0])
        atomwise_overlap_lumo1 = np.zeros(coord1.shape[0])
        atomwise_overlap_lumo2 = np.zeros(coord2.shape[0])


        hmmopipole_trans1 = hmmopipole @ R1.T
        hmmopipole_trans2 = hmmopipole @ R2.T
        hmmo_porbpop1 = hmmopipole_trans1.ravel()
        hmmo_porbpop2 = hmmopipole_trans2.ravel()
        pint_hmmo_frag1, pint_hmmo_frag2 = compute_atomwise_overlap_kernal(
            gtfcoef0, gtfcontr, gtfcenter1, gtflxyz, gtfatidx, 
            gtfcoef0, gtfcontr, gtfcenter2, gtflxyz, gtfatidx, 
            hmmo_porbpop1, hmmo_porbpop2,
        )

        lumopipole_trans1 = np.dot(R1, lumopipole.T).T
        lumopipole_trans2 = np.dot(R2, lumopipole.T).T
        lumo_porbpop1 = lumopipole_trans1.ravel()
        lumo_porbpop2 = lumopipole_trans2.ravel()
        pint_lumo_frag1, pint_lumo_frag2 = compute_atomwise_overlap_kernal(
            gtfcoef0, gtfcontr, gtfcenter1, gtflxyz, gtfatidx, 
            gtfcoef0, gtfcontr, gtfcenter2, gtflxyz, gtfatidx, 
            lumo_porbpop1, lumo_porbpop2,
        )

        for i, (v_hh, v_ll) in enumerate(zip(pint_hmmo_frag1, pint_lumo_frag1)):
            atomwise_overlap_hmmo1[i] = v_hh
            atomwise_overlap_lumo1[i] = v_ll
        for i, (v_hh, v_ll) in enumerate(zip(pint_hmmo_frag2, pint_lumo_frag2)):
            atomwise_overlap_hmmo2[i] = v_hh
            atomwise_overlap_lumo2[i] = v_ll

        atomwise_overlaps_hmmo1.append(atomwise_overlap_hmmo1)
        atomwise_overlaps_hmmo2.append(atomwise_overlap_hmmo2)
        atomwise_overlaps_lumo1.append(atomwise_overlap_lumo1)
        atomwise_overlaps_lumo2.append(atomwise_overlap_lumo2)

    return [atomwise_overlaps_hmmo1, atomwise_overlaps_hmmo2, atomwise_overlaps_lumo1, atomwise_overlaps_lumo2]

def compute_overlap_single_process_with_phase_correction(
    porb_pops: pd.DataFrame,
    ref_hmmo_ppop: pd.DataFrame,
    ref_lumo_ppop: pd.DataFrame,
    use_tqdm: bool = False,
    n_atom_per_monomer = None, 
    ):

    # retrieve the reference information
    ref_coord = ref_hmmo_ppop.groupby("aidx").mean().reset_index().loc[:, ("x", "y", "z")].values
    ref_center = np.mean(ref_coord, axis=0)

    ref_hmmoppop = np.array(ref_hmmo_ppop.loc[:,("px", "py", "pz")])
    ref_lumoppop = np.array(ref_lumo_ppop.loc[:,("px", "py", "pz")])
    gtfcoord = np.array(ref_hmmo_ppop.loc[:,("x", "y", "z")])
    gtfcoef0 = np.array(ref_hmmo_ppop.loc[:,"c"])
    gtfcontr = np.array(ref_hmmo_ppop.loc[:,"a"])
    gtfatidx = np.array(ref_hmmo_ppop.loc[:,"aidx"])
    gtfcoord = np.repeat(gtfcoord, 3, axis=0)
    gtfcoef0 = np.repeat(gtfcoef0, 3)
    gtfcontr = np.repeat(gtfcontr, 3)
    gtfatidx = np.repeat(gtfatidx, 3)
    gtflxyz = np.tile([[1,0,0],[0,1,0],[0,0,1]], reps = (ref_hmmo_ppop.shape[0], 1))
    
    n = porb_pops.index.levels[0].shape[0]
    atomwise_overlaps_hmmo1 = []
    atomwise_overlaps_hmmo2 = []
    atomwise_overlaps_lumo1 = []
    atomwise_overlaps_lumo2 = []

    imin = porb_pops.index.get_level_values(0).min()
    imax = porb_pops.index.get_level_values(0).max() + 1
    iterer = tqdm.tqdm(range(imin, imax)) if use_tqdm else range(imin, imax)
    for i in iterer:
        mn1_hmmo_ppop = porb_pops.loc[(i, "h1", slice(None)), :]
        mn2_hmmo_ppop = porb_pops.loc[(i, "h2", slice(None)), :]
        mn1_lumo_ppop = porb_pops.loc[(i, "l1", slice(None)), :]
        mn2_lumo_ppop = porb_pops.loc[(i, "l2", slice(None)), :]

        # get the translation and rotation matrix from porbpop
        mn1_coord = mn1_hmmo_ppop.groupby("aidx").mean().reset_index().loc[:, ("x", "y", "z")].values
        mn2_coord = mn2_hmmo_ppop.groupby("aidx").mean().reset_index().loc[:, ("x", "y", "z")].values
        mn1_center = np.mean(mn1_coord, axis=0)
        mn2_center = np.mean(mn2_coord, axis=0)
        R1 = get_transformation_matrix(ref_coord, mn1_coord)
        R2 = get_transformation_matrix(ref_coord, mn2_coord)

        # transform the reference gtf coordinates and p-population to aligh with monomer 1 and monomer 2
        trs_mn1_gtfcoord = ((gtfcoord - ref_center) @ R1.T + mn1_center)
        trs_mn2_gtfcoord = ((gtfcoord - ref_center) @ R2.T + mn2_center)
        trs_mn1_hmmoppop = (ref_hmmoppop @ R1.T).ravel()
        trs_mn2_hmmoppop = (ref_hmmoppop @ R2.T).ravel()
        trs_mn1_lumoppop = (ref_lumoppop @ R1.T).ravel()
        trs_mn2_lumoppop = (ref_lumoppop @ R2.T).ravel()

        # get the original phase of the monomer 1 and monomer 2, then correct the phase of the transformed p-population
        mn1_hmmoppop = np.array(mn1_hmmo_ppop.loc[:,("px", "py", "pz")]).ravel()
        mn2_hmmoppop = np.array(mn2_hmmo_ppop.loc[:,("px", "py", "pz")]).ravel()
        mn1_lumoppop = np.array(mn1_lumo_ppop.loc[:,("px", "py", "pz")]).ravel()
        mn2_lumoppop = np.array(mn2_lumo_ppop.loc[:,("px", "py", "pz")]).ravel()
        trs_mn1_hmmoppop *= np.sign(np.dot(trs_mn1_hmmoppop, mn1_hmmoppop))
        trs_mn2_hmmoppop *= np.sign(np.dot(trs_mn2_hmmoppop, mn2_hmmoppop))
        trs_mn1_lumoppop *= np.sign(np.dot(trs_mn1_lumoppop, mn1_lumoppop))
        trs_mn2_lumoppop *= np.sign(np.dot(trs_mn2_lumoppop, mn2_lumoppop))

        # initialize the array for storing the atomwise overlap
        atomwise_overlap_hmmo1 = np.zeros(mn1_coord.shape[0] if n_atom_per_monomer is None else n_atom_per_monomer)
        atomwise_overlap_hmmo2 = np.zeros(mn2_coord.shape[0] if n_atom_per_monomer is None else n_atom_per_monomer)
        atomwise_overlap_lumo1 = np.zeros(mn1_coord.shape[0] if n_atom_per_monomer is None else n_atom_per_monomer)
        atomwise_overlap_lumo2 = np.zeros(mn2_coord.shape[0] if n_atom_per_monomer is None else n_atom_per_monomer)

        pint_hmmo_frag1, pint_hmmo_frag2 = compute_atomwise_overlap_kernal(
            gtfcoef0, gtfcontr, trs_mn1_gtfcoord, gtflxyz, gtfatidx, 
            gtfcoef0, gtfcontr, trs_mn2_gtfcoord, gtflxyz, gtfatidx,
            trs_mn1_hmmoppop, trs_mn2_hmmoppop,
        )
        pint_lumo_frag1, pint_lumo_frag2 = compute_atomwise_overlap_kernal(
            gtfcoef0, gtfcontr, trs_mn1_gtfcoord, gtflxyz, gtfatidx, 
            gtfcoef0, gtfcontr, trs_mn2_gtfcoord, gtflxyz, gtfatidx,
            trs_mn1_lumoppop, trs_mn2_lumoppop,
        )
        for ii, (v_hh, v_ll) in enumerate(zip(pint_hmmo_frag1, pint_lumo_frag1)):
            atomwise_overlap_hmmo1[ii] = v_hh
            atomwise_overlap_lumo1[ii] = v_ll
        for ii, (v_hh, v_ll) in enumerate(zip(pint_hmmo_frag2, pint_lumo_frag2)):
            atomwise_overlap_hmmo2[ii] = v_hh
            atomwise_overlap_lumo2[ii] = v_ll

        atomwise_overlaps_hmmo1.append(atomwise_overlap_hmmo1)
        atomwise_overlaps_hmmo2.append(atomwise_overlap_hmmo2)
        atomwise_overlaps_lumo1.append(atomwise_overlap_lumo1)
        atomwise_overlaps_lumo2.append(atomwise_overlap_lumo2)

    return [atomwise_overlaps_hmmo1, atomwise_overlaps_hmmo2, atomwise_overlaps_lumo1, atomwise_overlaps_lumo2]

def compute_overlap_main(conformers, refmonomerwf:str, nproc:int=16, outputfolder:str=""):
    t = time.time()
    mwf = MoldenWavefunction(refmonomerwf)
    n_atom_monomer = mwf.molecule.coords.shape[0]
    mcoord = mwf.molecule.coords * 0.529177
    hmmopop = get_porb_pop(mwf, mwf.hmmo)
    lumopop = get_porb_pop(mwf, mwf.lumo)
    monomer_name = os.path.basename(refmonomerwf).split(".")[0]

    if outputfolder != "":
        os.makedirs(outputfolder, exist_ok=True)
        hmmopop.to_csv(f"{outputfolder}/{monomer_name}_hmmopop.csv")
        lumopop.to_csv(f"{outputfolder}/{monomer_name}_lumopop.csv")

    hmmopops = [hmmopop for i in range(nproc)]
    lumopops = [lumopop for i in range(nproc)]
    mcoordss = [mcoord for i in range(nproc)]

    use_wavefunction = False 
    if isinstance(conformers, str):
        traj_name = os.path.basename(conformers).split(".")[0]
        traj = md.load_hdf5(conformers)
        n_frame, n_atom = traj.n_frames, traj.n_atoms
        xyzs = traj.xyz * 10
    elif isinstance(conformers, np.ndarray):
        traj_name = "monomer"
        xyzs = conformers
        n_frame, n_atom = xyzs.shape[0], xyzs.shape[1]
    elif isinstance(conformers, pd.DataFrame):
        traj_name = "monomer"
        use_wavefunction = True
        n_frame = conformers.index.levels[0].shape[0]
        n_atom = n_atom_monomer * 2
        porb_pop = conformers
    elif isinstance(conformers, list):
        traj_name = "monomer"
        use_wavefunction = True
        n_frame = len(conformers)
        n_atom = n_atom_monomer * 2
        porb_pop = convert_molden_to_porb_pop(conformers)
        

    atomwise_overlaps_hmmo1 = []
    atomwise_overlaps_hmmo2 = []
    atomwise_overlaps_lumo1 = []
    atomwise_overlaps_lumo2 = []
    resultsss = [atomwise_overlaps_hmmo1, atomwise_overlaps_hmmo2, atomwise_overlaps_lumo1, atomwise_overlaps_lumo2]

    if not use_wavefunction: 
        xyzchunks = np.array_split(xyzs, nproc, axis=0)
        if nproc > 1:
            with Pool(nproc) as p:
                results = p.starmap(compute_overlap_single_process, zip(xyzchunks, hmmopops, lumopops, mcoordss))
            for i in range(nproc):
                for j in range(4):
                    resultsss[j].extend(results[i][j])
        else:
            results = compute_overlap_single_process(xyzs, hmmopop, lumopop, mcoord, use_tqdm=True)
            for i in range(4):
                resultsss[i].extend(results[i])
    elif use_wavefunction:
        index_chunks = np.array_split(porb_pop.index.levels[0].values, nproc)
        df_chunks = [porb_pop.loc[idxs].copy() for idxs in index_chunks]
        if nproc > 1:
            with Pool(nproc) as p:
                results = p.starmap(
                    compute_overlap_single_process_with_phase_correction, 
                    zip(df_chunks, hmmopops, lumopops, 
                        [False for i in range(nproc)], [n_atom_monomer for i in range(nproc)])
                        )
            for i in range(nproc):
                for j in range(4):
                    resultsss[j].extend(results[i][j])
        else:
            results = compute_overlap_single_process_with_phase_correction(
                porb_pop, hmmopop, lumopop, use_tqdm=True, n_atom_per_monomer = n_atom_monomer)
            for i in range(4):
                resultsss[i].extend(results[i])


    atomwise_overlaps_hmmo1 = np.array(resultsss[0])
    atomwise_overlaps_hmmo2 = np.array(resultsss[1])
    atomwise_overlaps_lumo1 = np.array(resultsss[2])
    atomwise_overlaps_lumo2 = np.array(resultsss[3])

    atomwise_overlaps_hmmo = np.concatenate((atomwise_overlaps_hmmo1, atomwise_overlaps_hmmo2), axis=1)
    atomwise_overlaps_lumo = np.concatenate((atomwise_overlaps_lumo1, atomwise_overlaps_lumo2), axis=1)

    df_hmmo = pd.DataFrame(index = range(n_atom), columns = range(n_frame), data = atomwise_overlaps_hmmo.T)
    df_lumo = pd.DataFrame(index = range(n_atom), columns = range(n_frame), data = atomwise_overlaps_lumo.T)
        
    if outputfolder != "":
        df_hmmo = pd.DataFrame(index = range(n_atom), columns = range(n_frame), data = atomwise_overlaps_hmmo.T)
        df_lumo = pd.DataFrame(index = range(n_atom), columns = range(n_frame), data = atomwise_overlaps_lumo.T)
        df_hmmo.to_csv(f"{outputfolder}/{traj_name}-ovlphmo.csv")
        df_lumo.to_csv(f"{outputfolder}/{traj_name}-ovlplmo.csv")

    print(f"Time elapsed: {time.time()-t:.4f} s")
    print(f"Time per frame: {(time.time()-t)/n_frame:.4f} s")
    print(f"Frame per second: {n_frame/(time.time()-t):.4f}")
    print(f"Core time per frame: {(time.time()-t)/n_frame*nproc:.4f} s")
    return df_hmmo, df_lumo




if __name__ == "__main__1":
    nproc = 20
    monomername     = "perylene" 
    monomerfolder   = "/home/fren5/asphaltene-ml/trimer-exmod/gathering_dataset/monomer" 
    outfolder       = "/home/fren5/asphaltene-ml/wavefunction-phasecorrect/outs" 
    moldenfolder    = "/home/fren5/asphaltene-ml/wavefunction-phasecorrect/moldens"

    ref_wf_path = f"{monomerfolder}/{monomername}.molden"
    conformers = "/home/fren5/asphaltene-ml/wavefunction-phasecorrect/data/perylene-test-porbpop.csv"
    conformers = pd.read_csv(conformers, index_col = [0,1,2])

    atmovlphmo, atmovlplmo = compute_overlap_main(conformers, ref_wf_path, nproc, outfolder)
    np.savez(f"{outfolder}/{monomername}-atmovl.npz", atmovlphmo=atmovlphmo, atmovlplmo=atmovlplmo)


if __name__ == "__main__":
    nproc = 64
    name = "tetracene-dimer-sep5A"
    monomername = "tetracene"
    conformers = f"/home/fren5/asphaltene-ml/trimer-exmod/gathering_dataset/data/{name}.h5"
    refmonomerwf = f"/home/fren5/asphaltene-ml/trimer-exmod/gathering_dataset/monomer/{monomername}.molden"
    outputfolder = os.getcwd()
    compute_overlap_main(conformers, refmonomerwf, nproc, outputfolder)

if __name__ == "__main__1":
    conformers = sys.argv[1]
    refmonomerwf = sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    outputfolder = sys.argv[4] if len(sys.argv) > 4 else ""
    atmovlphmo, atmpvlplmo = compute_overlap_main(conformers, refmonomerwf, nproc, outputfolder)

    traj_name = os.path.basename(conformers).split(".")[0]
    np.save(outputfolder + "/" + traj_name + "-" + "atmovlphmo.npy", atmovlphmo)
    np.save(outputfolder + "/" + traj_name + "-" + "atmovlplmo.npy", atmpvlplmo)
