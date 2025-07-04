import os
from multiprocessing import Pool, Process, Queue, Manager

import numpy as np
import mdtraj as md


def get_dimers(traj:md.Trajectory, threshold = 3.0, method = "com"):
    n_molecule = traj.top.n_residues
    n_frames = traj.n_frames
    molidxs_all = [traj.top.select(f"resid {i}") for i in range(n_molecule)]
    molidxs_heavy = [traj.top.select(f"resid {i} and type == C") for i in range(n_molecule)]

    mol_trajs = [traj.atom_slice(molidxs_heavy[i]) for i in range(n_molecule)]
    com_trajs = np.empty((n_molecule, traj.n_frames, 3))   
    for i in range(n_molecule):
        com_trajs[i] = md.compute_center_of_mass(mol_trajs[i])  
    com_trajs = com_trajs.reshape((n_molecule, n_frames, 1, 3))
    mol_trajs = np.array([mol_trajs[i].xyz for i in range(n_molecule)])
    mol_trajs = mol_trajs.transpose(1, 0, 2, 3) # (F, M, A, 3)
    com_trajs = com_trajs.transpose(1, 0, 2, 3) # (F, M, 1, 3)
    n_atoms = len(molidxs_all[0])

    # compute the distance matrix between molecules
    if method == "com":
        tD = np.min(np.linalg.norm(com_trajs[:, None, :, :, :] - com_trajs[:, :, None, :, :], axis = -1), axis = -1)
    elif method == "nearest":
        tD = np.min(np.linalg.norm(mol_trajs[:, None, :, :, :] - mol_trajs[:, :, None, :, :], axis = -1), axis = -1)
    elif method == "com2nearest":
        tD = np.min(np.linalg.norm(mol_trajs[:, None, :, :, :] - com_trajs[:, :, None, :, :], axis = -1), axis = -1)
    else:
        raise ValueError("method not recognized, can only be 'com', 'nearest', or 'com2nearest'")
    trilidx = np.tril_indices(n_molecule, k = 0)
    for i in range(n_frames):
        tD[i][trilidx] = np.inf

    # select the indices where the distance is smaller than the threshold
    idxs = np.argwhere(tD < threshold / 10.0)   # the threshold is in angstrom, while the distance matrix is in nm
    if len(idxs) == 0:
        return None

    del tD, com_trajs, mol_trajs
    mol_trajs_all = [traj.atom_slice(molidxs_all[i]) for i in range(n_molecule)]
    dimer_xyzs = np.empty((idxs.shape[0], n_atoms * 2, 3))
    for i, idx in enumerate(idxs):
        dimer_xyzs[i] = np.concatenate((mol_trajs_all[idx[1]][idx[0]].xyz, mol_trajs_all[idx[2]][idx[0]].xyz), axis = 1)

    # create a reference topology for storing the dimer
    reference_slice = traj.slice(0).atom_slice(np.concatenate((molidxs_all[idxs[0, 1]], molidxs_all[idxs[0, 2]]), axis = 0))

    return md.Trajectory(dimer_xyzs, reference_slice.top)


def get_dimers_single_process(chunk:md.Trajectory, threshold = 3.0, method = "com", startframe = 0, outputfolder = ".", fileprefix = "dimer"):
    n_frames = chunk.n_frames
    dimers = get_dimers(chunk, threshold = threshold, method = method)
    if dimers is None:
        print("No dimers found between frame {:>5d} and {:>5d}".format(startframe, startframe + n_frames), flush = True)
        return 
    print("Found {} dimers between frame {:>5d} and {:>5d}".format(len(dimers), startframe, startframe + n_frames), flush = True)
    if len(dimers) > 100000:
        dimers = dimers[np.random.permutation(len(dimers))[:100000]]
    dimers.save_hdf5(f"{outputfolder}/{fileprefix}-{startframe:>05d}-{startframe + n_frames:>05d}.h5")

 
def get_dimers_main(trajpath, toppath = None, threshold = 3.0, method = "com", chunksize = 100, n_proc = 16, outputfolder = ".", fileprefix = "dimer"):
    
    chunks = []
    startframes = []
    startframe = 0
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    for chunk in md.iterload(trajpath, chunk = chunksize, top = toppath):
        chunks.append(chunk)
        startframes.append(startframe)
        print("Loaded frame {:>5d} to {:>5d}".format(startframe, startframe + chunk.n_frames), flush = True)
        if len(chunks) >= n_proc:
            pool = Pool(processes = n_proc)
            pool.starmap(get_dimers_single_process, [(chunk, threshold, method, stf, outputfolder, fileprefix) for chunk, stf in zip(chunks, startframes)])
            pool.close()
            chunks = [] 
            startframes = []
        startframe += chunk.n_frames
        # if startframe >= 100:
        #     break

    if len(chunks) > 0:
        pool = Pool(processes = n_proc)
        pool.starmap(get_dimers_single_process, [(chunk, threshold, method, stf, outputfolder, fileprefix) for chunk, stf in zip(chunks, startframes)])
        pool.close()
        chunks = []
        startframes = []


import sys
if __name__ == "__main__":
    # specify the path to the trajectory and topology file
    trajpath        = "perylene-400-mmnpt.netcdf"
    toppath         = "perylene-400.prmtop"

    # Generate COM4A-undeduplicated
    threshold       = 4.0           
    method          = "com2nearest" 
    outputfolder    = "PrDim-COM4A-undeduplicated"
    chunksize       = 250           
    nproc           = 32            
    get_dimers_main(trajpath, toppath = toppath, threshold = threshold, method = method, chunksize = chunksize, n_proc = nproc, outputfolder = outputfolder, fileprefix = "dimer")

    # Generate NST5A-undeduplicated
    threshold       = 5.0           
    method          = "nearest"     
    outputfolder    = "PrDim-NST5A-undeduplicated"
    chunksize       = 250           
    nproc           = 32            
    get_dimers_main(trajpath, toppath = toppath, threshold = threshold, method = method, chunksize = chunksize, n_proc = nproc, outputfolder = outputfolder, fileprefix = "dimer")

