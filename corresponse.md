## Term correspondence

### General rule
 - out of bracket numbers indicates monomer index. `1` means monomer A, `2` means monomer B.
 - in bracket numbers indicates LE excited state index. `(1)` means the first monomer excited state, `(2)` means the second monomer excited state. The terms higher than `(1)` are not used in this study. 

### Monomeric properties

| Term in code | Term in article | Explanation |
| ------------- | ---------------- | ---------------- |
| `1e(1)g` | $E_{LE(gas)}^{A(1)}$ | The first excited state energy of monomer A in the gas phase. (isolated monomer A) |
| `1ip` | $IP_{A}$ | The ionization potential of isolated monomer A. |
| `2ea` | $EA_{B}$ | The electron affinity of isolated monomer B. |

### Diagonal terms
| Term in code | Term in article | Explanation |
| ------------- | ---------------- | ---------------- |
| `1e(1)` | $E_{LE}^{A(1)}$ | The first excited state energy of monomer A with the presence of the other monomer ($\|\Psi_{LE}^{A(1)} \rangle $) |
| `1+2-` | $E_{CT}^{A \rightarrow B}$ | The charge transfer energy with A donating an electron to B ($\|\Psi_{CT}^{A \rightarrow B} \rangle $) |

### Off-diagonal terms
| Term in code | Term in article | Explanation |
| ------------- | ---------------- | ---------------- |
| `1e(1)2e(1)` | $V_{LE-LE}^{A(1)B(1)}$ | The LE-LE coupling between the first excited states of monomers A and B ($\|\Psi_{LE}^{A(1)} \rangle $ and $\|\Psi_{LE}^{B(1)} \rangle $). |
| `1e(1)1+2-` | $V_{LE-CT}^{A(1)A \rightarrow B}$ | The LE-CT coupling between the first excited state of monomer A and the charge transfer state where A donates an electron to B. ($\|\Psi_{LE}^{A(1)} \rangle $ and $\|\Psi_{CT}^{A \rightarrow B} \rangle $). |
| `1e(1)1-2+` | $V_{LE-CT}^{A(1)B \rightarrow A}$ | The LE-CT coupling between the first excited state of monomer A and the charge transfer state where B donates an electron to A. ($\|\Psi_{LE}^{A(1)} \rangle $ and $\|\Psi_{CT}^{B \rightarrow A} \rangle $). |


### Approximation terms
| Term in code | Term in article | Explanation |
| ------------- | ---------------- | ---------------- |
| `atmapx_v_1+2-` | $V_{RESP}^{A+B-}$ | approximation of the Coulombic interaction between cationic A and anionic B, usint the Coulombic interaction between a fixed set of atomic RESP charges. |
| `atmapx_1e(1)2e(1)` | $V_{TrESP}^{A(1)B(1)}$ | approximation of the LE-LE coupling between the first excited states of monomers A and B, using the Coulombic interaction between a fixed set of atomic TrESP charges. |
| `atmapx_1e(1)1+2-` | $S_{LUMO}^{AB}$ | approximation of the LE-CT coupling between the first excited state of monomer A and the charge transfer state where B donates an electron to A, using the overlap integral between the LUMO of A and the LUMO of B. |
| `atmapx_1e(1)1-2+` | $S_{HOMO}^{AB}$ | approximation of the LE-CT coupling between the first excited state of monomer A and the charge transfer state where A donates an electron to B, using the overlap integral between the HOMO of B and the HOMO of A. |