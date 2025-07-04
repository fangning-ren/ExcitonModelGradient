import mdtraj as md
import numpy as np
import os
from multiprocessing import Pool
from typing import List


def deduplicate1(traj:md.Trajectory, rmsd_threshold:float, parallel = False):
    """
    Remove duplicate frames from a trajectory
    """
    n_frame = traj.n_frames
    traj = traj.center_coordinates()
    for i in range(n_frame):
        if i >= traj.n_frames:
            break
        rmsds = md.rmsd(traj, traj, i, precentered=True, parallel=parallel)
        rmsds[i] = np.inf
        nframe0 = traj.n_frames
        unique_frames = np.where(rmsds > rmsd_threshold)[0]
        traj = traj.slice(unique_frames, copy=False)
        if i % 100 == 0:
            print(f"Frame {i}, {traj.n_frames} / {n_frame} frames left")
    print(f"Unique frames: {traj.n_frames} / {n_frame}")
    return traj

def deduplicate(traj:md.Trajectory, rmsd_threshold:float, parallel = False):
    global n_samples
    cfms = traj.center_coordinates()
    n_points = cfms.n_frames
    min_distance_to_sampled = np.ones(n_points) * np.inf
    farthest = 0
    indices = [farthest,]
    for k in range(1, n_samples):
        farthest = indices[-1]
        distances = md.rmsd(cfms, cfms[farthest], precentered=True, parallel=parallel)
        min_distance_to_sampled = np.minimum(min_distance_to_sampled, distances)
        farthest = np.argmax(min_distance_to_sampled)
        indices.append(farthest)
        min_distance_to_sampled[farthest] = 0.0
        if k % 100 == 0:
            print(f"Frame {k} done")
        
    return cfms[indices]

def deduplicate_serial(trajfilestr:str, rmsd_threshold):
    return deduplicate(md.load_hdf5(trajfilestr), rmsd_threshold)

def deduplicate_parallel(trajfilepaths:List[str], rmsd_threshold:float, nproc:int = 1):
    """
    Remove duplicate frames from a trajectory in parallel
    """
    pool = Pool(nproc)
    async_results = []
    for trajfile in trajfilepaths:
        async_results.append(pool.apply_async(deduplicate_serial, (trajfile, rmsd_threshold)))
    pool.close()
    pool.join()
    trajs = [async_result.get() for async_result in async_results]
    traj = deduplicate(md.join(trajs, check_topology=True), rmsd_threshold=rmsd_threshold, parallel=True)
    return traj



if __name__ == "__main__":
    trajpath = "PrDim-COM4A-undeduplicated"  # Path to the directory containing the trajectory files, should be mdtraj generated hdf5 format
    trajfiles = [os.path.join(trajpath, f) for f in os.listdir(trajpath) if f.endswith(".h5") and f.startswith("dimer")]
    trajfiles.sort()

    n_samples = 8000
    traj = deduplicate_parallel(trajfiles, rmsd_threshold=0.05, nproc=21)
    traj.save_hdf5(os.path.join(trajpath, "PrDim-COM4A.h5"))