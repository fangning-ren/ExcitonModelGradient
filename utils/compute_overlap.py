from math import factorial
import numpy as np
from numba import njit
from multiprocessing import Pool, Process, Queue

from .molden_convert import *

def compute_pre_ext(a1, a2, rab, gam):
    return np.exp(-a1 * a2 * rab**2 / gam)

def compute_C(l, m):
    return factorial(l) / factorial(l-m) / factorial(m)

def bifactorial(k):
    return 1 if k <= 1 else k * bifactorial(k-2)
Bifactorials = np.array([bifactorial(k) for k in range(2*8+1)])
Bifactorials[-1] = 1
IntxCoeffs = np.array([np.pi**0.5 * Bifactorials[2*k-1] / 2**k for k in range(0, 8)])
IntxCoeffs = np.array([1.77245385e+00,8.86226925e-01,1.32934039e+00,3.32335097e+00,1.16317284e+01,5.23427778e+01,2.87885278e+02,1.87125431e+03])

Combination_numbers = np.array([
    [1, 1, 1, 1, 1],
    [0, 1, 2, 3, 4],
    [0, 0, 1, 3, 6],
    [0, 0, 0, 1, 4],
    [0, 0, 0, 0, 1]
], dtype = np.float64).T

@njit
def compute_f(i, l1, l2, rpa, rpb):
    # s, p, d, f, g: L = 0, 1, 2, 3, 4
    f = 0.0
    lui, luj = (l1, i-l1) if i > l1 else (i, 0)
    rui, ruj = (i-l2, l2) if i > l2 else (0, i)
    for k in range(0, ruj-luj+1):
        m, n = lui-k, luj+k
        f += rpa**(l1-m) * rpb**(l2-n) * Combination_numbers[l1, m] * Combination_numbers[l2, n]
    return f


@njit
def compute_Ix_0(l1, l2, rpa, rpb, gam):
    I = 0.0
    for i in range((l1 + l2)+1):
        if i % 2 == 0:
            k = i // 2
            I += compute_f(i, l1, l2, rpa, rpb) * IntxCoeffs[k] * gam**(-k - 0.5)
    return I

@njit
def compute_Ix(l1, l2, rpa, rpb, gam):
    I = 0.0
    for k in range((l1 + l2 + 1) // 2 + 1):
        I += compute_f(k*2, l1, l2, rpa, rpb) * IntxCoeffs[k] * gam**(-k - 0.5)
    return I

@njit
def compute_gtf_integral(c1, a1, r1, l1, c2, a2, r2, l2):
    rab = np.linalg.norm(r1 - r2)
    gam = a1 + a2
    rp = (a1 * r1 + a2 * r2) / gam
    xp, yp, zp = rp
    x1, y1, z1 = r1
    x2, y2, z2 = r2
    l1x, l1y, l1z = l1
    l2x, l2y, l2z = l2
    xpa, xpb = xp - x1, xp - x2
    ypa, ypb = yp - y1, yp - y2
    zpa, zpb = zp - z1, zp - z2
    s12 = c1 * c2 * np.exp(-a1 * a2 * rab**2 / gam) * compute_Ix(l1x, l2x, xpa, xpb, gam) * compute_Ix(l1y, l2y, ypa, ypb, gam) * compute_Ix(l1z, l2z, zpa, zpb, gam)
    return s12

@njit
def compute_S(
        c1:np.ndarray, a1:np.ndarray, r1:np.ndarray, l1:np.ndarray,
        c2:np.ndarray, a2:np.ndarray, r2:np.ndarray, l2:np.ndarray,
        startend1:np.ndarray, startend2:np.ndarray):
    S = np.zeros((c1.shape[0], c2.shape[0]), dtype = np.float64)
    for m in range(c1.shape[0]):
        for n in range(c2.shape[0]):
            S[m,n] += compute_gtf_integral(c1[m], a1[m], r1[m], l1[m], c2[n], a2[n], r2[n], l2[n])

    S1 = np.zeros((startend1.shape[0], startend2.shape[0]), dtype=np.float64)   
    for i in range(S1.shape[0]):
        for j in range(S1.shape[1]):
            S1[i,j] = np.sum(S[startend1[i,0]:startend1[i,1], startend2[j,0]:startend2[j,1]])
            # S[i,j] += compute_gtf_integral(c1[m], a1[m], r1[m], l1[m], c2[n], a2[n], r2[n], l2[n])
            # print(i, j, m, n)
    return S1

@njit
def compute_orbital_overlap_kernal(
    c1:np.ndarray, a1:np.ndarray, r1:np.ndarray, l1:np.ndarray,
    c2:np.ndarray, a2:np.ndarray, r2:np.ndarray, l2:np.ndarray,
    orbgtfcoeff1:np.ndarray, orbgtfcoeff2:np.ndarray):
    # compute the overlap integral between two orbitals
    # do not compute the whole atomic basis overlap matrix
    threshold = 1e-6
    sss = 0.0
    for i in range(c1.shape[0]):
        if abs(orbgtfcoeff1[i]) < threshold:
            continue
        for j in range(c2.shape[0]):
            if abs(orbgtfcoeff2[j] * orbgtfcoeff1[i]) < threshold:
                continue
            estimation = np.max(np.abs(r1[i] - r2[j])) * min(a1[i], a2[j])
            # print(f"orbital {i} {j} estimation: {estimation}")
            if estimation > 6.0:
                continue
            sss += orbgtfcoeff1[i] * orbgtfcoeff2[j] * compute_gtf_integral(
                c1[i], a1[i], r1[i], l1[i], c2[j], a2[j], r2[j], l2[j])
    return sss

@njit
def compute_atomwise_overlap_kernal(
    c1:np.ndarray, a1:np.ndarray, r1:np.ndarray, l1:np.ndarray, aidx1:np.ndarray,
    c2:np.ndarray, a2:np.ndarray, r2:np.ndarray, l2:np.ndarray, aidx2:np.ndarray,
    orbgtfcoeff1:np.ndarray, orbgtfcoeff2:np.ndarray):
    # compute the overlap integral between two orbitals
    # summarize the contribution from different atoms. 
    aidx1 = aidx1 - 1
    aidx2 = aidx2 - 1
    atomwise_overlap1 = np.zeros(aidx1.max()+1, dtype = np.float64)
    atomwise_overlap2 = np.zeros(aidx2.max()+1, dtype = np.float64)
    threshold = 1e-6
    for i in range(c1.shape[0]):
        if abs(orbgtfcoeff1[i]) < threshold:
            continue
        for j in range(c2.shape[0]):
            if abs(orbgtfcoeff2[j] * orbgtfcoeff1[i]) < threshold:
                continue
            estimation = np.max(np.abs(r1[i] - r2[j])) * min(a1[i], a2[j])
            # print(f"orbital {i} {j} estimation: {estimation}")
            if estimation > 6.0:
                continue
            ovlp = orbgtfcoeff1[i] * orbgtfcoeff2[j] * compute_gtf_integral(
                c1[i], a1[i], r1[i], l1[i], c2[j], a2[j], r2[j], l2[j])
            atomwise_overlap1[aidx1[i]] += ovlp
            atomwise_overlap2[aidx2[j]] += ovlp
    return atomwise_overlap1, atomwise_overlap2

def compute_orbital_overlap(wf1:MoldenWavefunction, wf2:MoldenWavefunction, orbidx1:int, orbidx2:int):
    # compute the overlap integral between two orbitals
    # do not compute the whole atomic basis overlap matrix
    # wf1, wf2: MoldenWavefunction
    # orbidx1, orbidx2: int
    startend1 = get_gto_gtf_index(wf1)
    startend2 = get_gto_gtf_index(wf2)
    c1, a1, r1, l1, _ = wf1.get_raveled_gtf()
    c2, a2, r2, l2, _ = wf2.get_raveled_gtf()
    orbgtfcoeff1 = np.zeros(c1.shape[0], dtype = np.float32)
    orbgtfcoeff2 = np.zeros(c2.shape[0], dtype = np.float32)
    for i, (start, end) in enumerate(startend1):
        orbgtfcoeff1[start:end] = wf1.C[i, orbidx1]
    for i, (start, end) in enumerate(startend2):
        orbgtfcoeff2[start:end] = wf2.C[i, orbidx2]
    sss = compute_orbital_overlap_kernal(
        c1, a1, r1, l1, c2, a2, r2, l2, orbgtfcoeff1, orbgtfcoeff2)
    return sss

def compute_orbital_overlap_mpi(wf1:MoldenWavefunction, wf2:MoldenWavefunction, orbidx1:int, orbidx2:int, nproc = 16):
    # compute the overlap integral between two orbitals
    # do not compute the whole atomic basis overlap matrix
    # wf1, wf2: MoldenWavefunction
    # orbidx1, orbidx2: int
    startend1 = get_gto_gtf_index(wf1)
    startend2 = get_gto_gtf_index(wf2)
    c1, a1, r1, l1, _ = wf1.get_raveled_gtf()
    c2, a2, r2, l2, _ = wf2.get_raveled_gtf()
    orbgtfcoeff1 = np.zeros(c1.shape[0], dtype = np.float32)
    orbgtfcoeff2 = np.zeros(c2.shape[0], dtype = np.float32)
    for i, (start, end) in enumerate(startend1):
        orbgtfcoeff1[start:end] = wf1.C[i, orbidx1]
    for i, (start, end) in enumerate(startend2):
        orbgtfcoeff2[start:end] = wf2.C[i, orbidx2]

    # split the input data for parallel computing
    ngtf = c1.shape[0]
    ngtf_per_proc = (ngtf + nproc - 1) // nproc
    async_results = []
    pool = Pool(nproc)
    for i in range(nproc):
        start = i * ngtf_per_proc
        end = min((i + 1) * ngtf_per_proc, ngtf)
        print(f"start {start} end {end}")
        async_results.append(pool.apply_async(
            compute_orbital_overlap_kernal,
            args = (c1[start:end], a1[start:end], r1[start:end], l1[start:end],
                    c2, a2, r2, l2, orbgtfcoeff1[start:end], orbgtfcoeff2)))
    pool.close()
    pool.join()
    sss = 0.0
    for async_result in async_results:
        sss += async_result.get()
    return sss
    

def compute_S12(gto1:GTO, gto2:GTO):
    s12 = 0.0
    for I, gtf1 in enumerate(gto1.funcs):
        gtf1:GTF
        r1 = gtf1.p
        x1, y1, z1 = gtf1.p
        c1, a1 = gtf1.c, gtf1.a
        l1x, l1y, l1z = gtf1.i, gtf1.j, gtf1.k
        for J, gtf2 in enumerate(gto2.funcs):
            r2 = gtf2.p
            x2, y2, z2 = gtf2.p
            c2, a2 = gtf2.c, gtf2.a
            l2x, l2y, l2z = gtf2.i, gtf2.j, gtf2.k

            rab = np.linalg.norm(r1 - r2)
            gam = a1 + a2
            rp = (a1 * r1 + a2 * r2) / gam
            xp, yp, zp = rp
            xpa, xpb = xp - x1, xp - x2
            ypa, ypb = yp - y1, yp - y2
            zpa, zpb = zp - z1, zp - z2
            s12 += c1 * c2 * np.exp(-a1 * a2 * rab**2 / gam) * compute_Ix(l1x, l2x, xpa, xpb, gam) * compute_Ix(l1y, l2y, ypa, ypb, gam) * compute_Ix(l1z, l2z, zpa, zpb, gam)
    return s12   

def get_gto_gtf_index(wf:MoldenWavefunction):
    startends = np.empty((len(wf.gtos), 2), dtype = np.int64)
    cstart = 0
    for i, gto in enumerate(wf.gtos):
        gto:GTO
        startends[i,0] = cstart
        startends[i,1] = cstart + len(gto.funcs)
        cstart = startends[i,1]
    return startends

def compute_overlap_matrix(wf:MoldenWavefunction):
    nbasis = len(wf.gtos)
    S = np.empty((nbasis, nbasis), dtype = np.float64)
    for i in range(nbasis):
        for j in range(i, nbasis):
            S[i,j] = compute_S12(wf.gtos[i], wf.gtos[j])
            S[j,i] = S[i,j]
    return S

def compute_overlap_between_basis(wf1:MoldenWavefunction, wf2:MoldenWavefunction):
    ngto1, ngto2 = len(wf1.gtos), len(wf2.gtos)
    c1, a1, r1, l1, _ = wf1.get_raveled_gtf()
    c2, a2, r2, l2, _ = wf2.get_raveled_gtf()
    startend1 = get_gto_gtf_index(wf1)
    startend2 = get_gto_gtf_index(wf2)
    S = compute_S(c1, a1, r1, l1, c2, a2, r2, l2, startend1, startend2)
    return S

def compute_orbital_correspond_matrix(wf1:MoldenWavefunction, wf2:MoldenWavefunction, range1, range2):
    S = compute_overlap_between_basis(wf1, wf2)
    return wf1.C[:,range1[0]:range1[1]].T @ S @ wf2.C[:,range2[0]:range2[1]]


def main():
    wf = MoldenWavefunction(r"b25-b25-full/batch_06/scr-b25-b25-06-06/b25-b25-06-06.molden")
    S = compute_overlap_between_basis(wf, wf)
    np.savetxt("S.txt", S, fmt = "%22.18e")

# wf1 = MoldenWavefunction(r"result_monomer\absorption\moldens\coal09.molden")
# wf2 = MoldenWavefunction(r"result_monomer\absorption\moldens\coal10.molden")
# A = compute_orbital_correspond_matrix(wf1, wf2, [60, 70], [60, 70])
# print(A)

if __name__ == "__main__":
    import cProfile
    cProfile.run("main()")
