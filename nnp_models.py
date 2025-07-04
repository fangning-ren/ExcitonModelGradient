import numpy as np
import torch
import torchani
from torchani.aev import AEVComputer


class ANINNP(torch.nn.Module):

    def __init__(self, 
                dimer_aev = True, 
                modified_aev = False,
                use_belonging = True, 
                elements = ["H", "C", "N", "O"], 
                hidden_channels = [64, 32, 16],
                approx_channels = 1,
                output_channels = 1,
                Rcr = 5.2000e+00, EtaR = 1.6000000e+01, 
                ShfR_min = 0.9, ShfR_n = 16,
                Rca = 3.5000e+00, EtaA = 8.0000000e+00,
                ShfA_min = 0.9, ShfA_n = 4 ,
                Zeta = 3.2000000e+01, ShfZ_n = 8,
                device = "cpu",
        ):
        super(ANINNP, self).__init__()
        self.elements = elements
        self.num_species = len(elements)
        self.Rcr  = Rcr
        self.EtaR = torch.Tensor([EtaR,]).to(device)
        self.ShfR = torch.linspace(ShfR_min, Rcr-(Rcr-ShfR_min)/ShfR_n, ShfR_n).to(device)
        self.Rca  = Rca
        self.EtaA = torch.Tensor([EtaA]).to(device)
        self.ShfA = torch.linspace(ShfA_min, Rca-(Rca-ShfA_min)/ShfA_n, ShfA_n).to(device)
        self.Zeta = torch.Tensor([Zeta]).to(device)
        self.ShfZ = torch.linspace(torch.pi/ShfZ_n/2, torch.pi-torch.pi/ShfZ_n/2, ShfZ_n).to(device)
        self.aev_computer = AEVComputer(
            self.Rcr, self.Rca, self.EtaR, self.ShfR, 
            self.EtaA, self.Zeta, self.ShfA, self.ShfZ, 
            self.num_species * (2 if dimer_aev else 1))
        self.aev_dim = self.aev_computer.aev_length
        if use_belonging and not dimer_aev:
            self.aev_dim += 1               # belonging label input as an additional feature
        self.dimer_aev = dimer_aev
        self.modified_aev = modified_aev
        self.use_belonging = use_belonging
        self.device = device

        self.hidden_channels = hidden_channels
        self.approx_channels = approx_channels
        self.output_channels = output_channels
    
        element_network_sequence = []
        element_network_sequence.append(torch.nn.Linear(self.aev_dim + self.approx_channels, hidden_channels[0]))
        element_network_sequence.append(torch.nn.CELU(0.1))
        for j in range(len(hidden_channels)-1):
            element_network_sequence.append(torch.nn.Linear(hidden_channels[j], hidden_channels[j+1]))
            element_network_sequence.append(torch.nn.CELU(0.1))
        element_network_sequence.append(torch.nn.Linear(hidden_channels[-1], output_channels))
        self.element_network = torch.nn.Sequential(*element_network_sequence)
        # print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")
        # print(f"number of aev dimensions: {self.aev_dim}")
        # print(f"number of approx channels: {self.approx_channels}")
        # print(f"number of output channels: {self.output_channels}")

    def get_all_linear_layers(self):
        linear_layers = []
        for layer in self.element_network:
            if isinstance(layer, torch.nn.Linear):
                linear_layers.append(layer)
        return linear_layers

    def init_params(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        for layer in self.element_network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, species:torch.Tensor, coordinates:torch.Tensor, belongings:torch.Tensor = None, approx:torch.Tensor = None):
        # species = torch.Tensor(species).to(torch.int64)
        # coordinates = torch.Tensor(coordinates)

        if belongings is None or not self.use_belonging:
            _, aev = self.aev_computer((species, coordinates))
        elif self.use_belonging and self.dimer_aev and self.modified_aev:
            belongings_ = belongings - 1        # 0 = self, 1 = other   # (N, A)
            species_with_belonging_1 = species + belongings_ * self.num_species
            _, aev1 = self.aev_computer((species_with_belonging_1, coordinates))

            delongings_ = belongings.max() - belongings # 0 = self, 1 = other
            species_with_belonging_2 = species + delongings_ * self.num_species
            _, aev2 = self.aev_computer((species_with_belonging_2, coordinates))

            aev = aev1.new_zeros(aev1.shape)    # (N, A, aev_dim)
            aev[belongings_ == 0] = aev1[belongings_ == 0]  # until here no gradient is required so in-place operations are allowed 
            aev[delongings_ == 0] = aev2[delongings_ == 0]  # until here no gradient is required so in-place operations are allowed 
        elif self.use_belonging and self.dimer_aev:
            species_with_belonging = species + (belongings - 1) * self.num_species
            _, aev = self.aev_computer((species_with_belonging, coordinates))
        elif self.use_belonging:
            _, aev = self.aev_computer((species, coordinates))
            belongings_ = belongings * 2 - 3  # 1 -> -1, 2 -> 1
            aev = torch.cat([aev, belongings.unsqueeze(-1).to(aev.dtype)], dim = -1)
        else:
            _, aev = self.aev_computer((species, coordinates))

        if self.approx_channels > 0:
            if approx is None:
                approx = torch.zeros_like(aev[..., 0]).to(self.device)
            approx = approx.unsqueeze(-1) if approx.dim() == 2 else approx  # (N, A) to (N, A, 1) or (N, A, approx_channels)
            approx = approx.to(aev.dtype)
            aev = torch.cat([aev, approx], dim = -1)

        # adapted from torchani.nn.ANIModel
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros((species_.shape[0], self.output_channels))

        mask = torch.ones_like(species_, dtype=torch.bool, device=self.device)
        midx = mask.nonzero().flatten()
        if midx.shape[0] > 0:
            input_ = aev.index_select(0, midx)
            output_ = self.element_network(input_)
            output.index_add_(0, midx, output_)
        output = output.view((*species.shape, self.output_channels))

        return output.sum(dim = 1)


class LocalExcitedStateNNP(torch.nn.Module):

    def __init__(self, 
                device = "cpu", 
                monomer_nnp_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 1,
                    "approx_channels": 0,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                interaction_nnp_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 1,
                    "approx_channels": 1,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                dimer_aev     = True,
                modified_aev  = False,
                use_belonging = True, 
                output_1e1g   = False,
        ):
        super(LocalExcitedStateNNP, self).__init__()
        self.device = device
        print("Initializing monomer models")
        monomer_nnp_params = monomer_nnp_params.copy()
        monomer_nnp_params.update({"device": device, "dimer_aev": False, "modified_aev": False, "use_belonging": False})
        self.monomer_model = ANINNP(**monomer_nnp_params)
        print("Initializing interaction models")
        interaction_nnp_params = interaction_nnp_params.copy()
        interaction_nnp_params.update({"device": device, "dimer_aev": dimer_aev, "modified_aev": modified_aev, "use_belonging": use_belonging})
        self.interaction_model = ANINNP(**interaction_nnp_params)
        self.output_1e1g = output_1e1g

    def get_all_linear_layers(self):
        return self.monomer_model.get_all_linear_layers() + self.interaction_model.get_all_linear_layers()
    
    def init_params(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.monomer_model.init_params(seed)
        self.interaction_model.init_params(seed)
    
    def forward(self, species:torch.Tensor, coordinates:torch.Tensor, belongings:torch.Tensor, approx:torch.Tensor = None, edge_index = None, batch = None):
        """
        species:    (N, A)    tensor, where N is the number of molecules, A is the number of atoms in each molecule
        coordinates:(N, A, 3) tensor, where N is the number of molecules, A is the number of atoms in each molecule
        belongings: (N, A)    tensor, where N is the number of molecules, A is the number of atoms in each molecule
        approx:     (N, A, M) tensor, where N is the number of molecules, A is the number of atoms in each molecule, M is the number of approximations included
        """
        molecule_1_mask = belongings == 1
        molecule_2_mask = belongings == 2
        natom1 = molecule_1_mask[0].sum()
        natom2 = molecule_2_mask[0].sum()
        coord1_mask = molecule_1_mask.unsqueeze(-1).expand(coordinates.shape)
        coord1 = torch.masked_select(coordinates, coord1_mask).view(-1, natom1, 3)
        species1_mask = molecule_1_mask.expand(species.shape)
        species1 = torch.masked_select(species, species1_mask).view(-1, natom1)
        ele1g = self.monomer_model(species1, coord1)
        einta = self.interaction_model(species, coordinates, belongings, approx)
        if self.output_1e1g:
            return ele1g, ele1g + einta
        else:
            return ele1g + einta

class ChargeTransferStateNNP(torch.nn.Module):

    def __init__(self, 
                device = "cpu", 
                monomer_nnp_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 1,
                    "approx_channels": 0,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                interaction_nnp_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 1,
                    "approx_channels": 1,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                dimer_aev     = True,
                modified_aev  = False,
                use_belonging = True, 
                output_ipea   = False,
        ):
        super(ChargeTransferStateNNP, self).__init__()
        self.device = device
        monomer_nnp_params = monomer_nnp_params.copy()
        monomer_nnp_params.update({"device": device, "dimer_aev": False, "modified_aev": False, "use_belonging": False})
        interaction_nnp_params = interaction_nnp_params.copy()
        interaction_nnp_params.update({"device": device, "dimer_aev": dimer_aev, "modified_aev": modified_aev, "use_belonging": use_belonging})
        print("Initializing cation models")
        self.cation_model = ANINNP(**monomer_nnp_params)
        print("Initializing anion models")
        self.anion_model = ANINNP(**monomer_nnp_params)
        print("Initializing interaction models")
        self.interaction_model = ANINNP(**interaction_nnp_params)
        self.output_ipea = output_ipea

    def get_all_linear_layers(self):
        return self.cation_model.get_all_linear_layers() + self.anion_model.get_all_linear_layers() + self.interaction_model.get_all_linear_layers()
    
    def init_params(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.cation_model.init_params(seed)
        self.anion_model.init_params(seed)
        self.interaction_model.init_params(seed)
    
    def forward(self, species:torch.Tensor, coordinates:torch.Tensor, belongings:torch.Tensor, approx:torch.Tensor = None, edge_index = None, batch = None):
        """
        species:    (N, A)    tensor, where N is the number of molecules, A is the number of atoms in each molecule
        coordinates:(N, A, 3) tensor, where N is the number of molecules, A is the number of atoms in each molecule
        belongings: (N, A)    tensor, where N is the number of molecules, A is the number of atoms in each molecule
        approx:     (N, A, M) tensor, where N is the number of molecules, A is the number of atoms in each molecule, M is the number of approximations included
        """
        molecule_1_mask = belongings == 1
        molecule_2_mask = belongings == 2

        natom1 = molecule_1_mask[0].sum()
        natom2 = molecule_2_mask[0].sum()

        coord1_mask = molecule_1_mask.unsqueeze(-1).expand(coordinates.shape)
        coord1 = torch.masked_select(coordinates, coord1_mask).view(-1, natom1, 3)
        coord2_mask = molecule_2_mask.unsqueeze(-1).expand(coordinates.shape)
        coord2 = torch.masked_select(coordinates, coord2_mask).view(-1, natom2, 3)
        
        species1_mask = molecule_1_mask.expand(species.shape)
        species1 = torch.masked_select(species, species1_mask).view(-1, natom1)
        species2_mask = molecule_2_mask.expand(species.shape)
        species2 = torch.masked_select(species, species2_mask).view(-1, natom2)

        cation_energies = self.cation_model(species1, coord1)
        anion_energies = self.anion_model(species2, coord2)
        interaction_energies = self.interaction_model(species, coordinates, belongings, approx)

        if self.output_ipea:
            return cation_energies, anion_energies, interaction_energies + cation_energies + anion_energies
        else:
            return interaction_energies + cation_energies + anion_energies

class CouplingNNP(torch.nn.Module):

    def __init__(self, 
                device = "cpu", 
                model_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 1,
                    "approx_channels": 1,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                dimer_aev     = True,
                modified_aev  = False,
                use_belonging = True, 
                unsqueeze_output = False, 
                ):
        super(CouplingNNP, self).__init__()
        self.device = device

        model_params = model_params.copy()
        model_params.update({"device": device, "dimer_aev": dimer_aev, "modified_aev": modified_aev, "use_belonging": use_belonging})
        self.ani_model = ANINNP(**model_params)
        self.unsqueeze_output = unsqueeze_output

    def get_all_linear_layers(self):
        return self.ani_model.get_all_linear_layers()
    
    def init_params(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.ani_model.init_params(seed)
    
    def forward(self, species:torch.Tensor, coordinates:torch.Tensor, belongings:torch.Tensor, approx:torch.Tensor = None, edge_index = None, batch = None):
        """
        species:    (N, A)    tensor, where N is the number of molecules, A is the number of atoms in each molecule
        coordinates:(N, A, 3) tensor, where N is the number of molecules, A is the number of atoms in each molecule
        belongings: (N, A)    tensor, where N is the number of molecules, A is the number of atoms in each molecule
        approx:     (N, A, M) tensor, where N is the number of molecules, A is the number of atoms in each molecule, M is the number of approximations included
        """
        if not self.unsqueeze_output:
            return self.ani_model(species, coordinates, belongings, approx)
        else:
            predits = self.ani_model(species, coordinates, belongings, approx)  # (batch_size)
            return predits.unsqueeze(-1)  # (batch_size, 1)
 

class ExmodNNP(torch.nn.Module):

    def __init__(self, 
                device = "cpu", 
                monomer_nnp_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 3,
                    "approx_channels": 0,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                interaction_nnp_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 5,
                    "approx_channels": 4,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                dimer_aev     = True,
                modified_aev  = False,
                use_belonging = True, 
                return_monomer_terms = False, 
    ):
        super(ExmodNNP, self).__init__()
        self.device = device
        self.return_monomer_terms = return_monomer_terms

        print("Initializing monomer models")
        monomer_nnp_params = monomer_nnp_params.copy()
        monomer_nnp_params.update({"device": device, "dimer_aev": False, "modified_aev": False, "use_belonging": False})
        self.monomer_model = ANINNP(**monomer_nnp_params)

        print("Initializing interaction models")
        interaction_nnp_params = interaction_nnp_params.copy()
        interaction_nnp_params.update({"device": device, "dimer_aev": dimer_aev, "modified_aev": modified_aev, "use_belonging": use_belonging})
        self.interaction_model = ANINNP(**interaction_nnp_params)

    def init_params(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.monomer_model.init_params(seed)
        self.interaction_model.init_params(seed)
    
    def forward(self, species:torch.Tensor, coordinates:torch.Tensor, belongings:torch.Tensor, approx:torch.Tensor = None):
        """
        species: (N, A) tensor, where N is the number of molecules, A is the number of atoms in each molecule
        coordinates: (N, A, 3) tensor, where N is the number of molecules, A is the number of atoms in each molecule
        belongings: (N, A) tensor, where N is the number of molecules, A is the number of atoms in each molecule
        approx: (N, A, M) tensor, where N is the number of molecules, A is the number of atoms in each molecule, M is the number of approximations included
        output: (N, P) tensor, where N is the number of molecules, P is the number of properties to predict
        """
        molecule_1_mask = belongings == 1
        molecule_2_mask = belongings == 2

        natom1 = molecule_1_mask[0].sum()
        natom2 = molecule_2_mask[0].sum()

        coord1_mask = molecule_1_mask.unsqueeze(-1).expand(coordinates.shape)
        coord1 = torch.masked_select(coordinates, coord1_mask).view(-1, natom1, 3)
        coord2_mask = molecule_2_mask.unsqueeze(-1).expand(coordinates.shape)
        coord2 = torch.masked_select(coordinates, coord2_mask).view(-1, natom2, 3)
        
        species1_mask = molecule_1_mask.expand(species.shape)
        species1 = torch.masked_select(species, species1_mask).view(-1, natom1)
        species2_mask = molecule_2_mask.expand(species.shape)
        species2 = torch.masked_select(species, species2_mask).view(-1, natom2)

        monomer1_energies = self.monomer_model(species1, coord1)
        monomer2_energies = self.monomer_model(species2, coord2)
        interaction_terms = self.interaction_model(species, coordinates, belongings, approx)

        e_1g1 = monomer1_energies[:,0]
        e_ip1 = monomer1_energies[:,1]
        e_ea2 = monomer2_energies[:,2]
        e_1e1  = monomer1_energies[:,0] + interaction_terms[:,0]
        e_1p2m = monomer1_energies[:,1] + monomer2_energies[:,2] + interaction_terms[:,1]
        v_1e12e1  = interaction_terms[:,2]
        v_1e11p2m = interaction_terms[:,3]
        v_1e11m2p = interaction_terms[:,4]

        if self.return_monomer_terms:
            return torch.stack([e_1g1, e_ip1, e_ea2, e_1e1, e_1p2m, v_1e12e1, v_1e11p2m, v_1e11m2p], dim = -1)
        else:
            return torch.stack([e_1e1, e_1p2m, v_1e12e1, v_1e11p2m, v_1e11m2p], dim = -1)


class ExmodSeparateNNP(torch.nn.Module):
    
    def __init__(self, 
                device = "cpu", 
                monomer_nnp_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 1,
                    "approx_channels": 0,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                interaction_nnp_params = {
                    "elements": ["H", "C", "N", "O"],
                    "hidden_channels": [64, 32, 16],
                    "output_channels": 1,
                    "approx_channels": 1,
                    "Rcr": 5.2000e+00, "EtaR": 1.6000000e+01, 
                    "ShfR_min": 0.9, "ShfR_n": 16,
                    "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
                    "ShfA_min": 0.9, "ShfA_n": 4 ,
                    "Zeta": 3.2000000e+01, "ShfZ_n": 8,
                },
                dimer_aev     = True,
                modified_aev  = False,
                use_belonging = True, 
                return_monomer_terms = False, 
    ):
        super(ExmodSeparateNNP, self).__init__()
        self.device = device
        self.monomer_nnp_params = monomer_nnp_params
        self.interaction_nnp_params = interaction_nnp_params
        self.return_monomer_terms = return_monomer_terms

        self.monomer_nnp_params["approx_channels"] = 0
        self.monomer_nnp_params["output_channels"] = 1
        self.interaction_nnp_params["output_channels"] = 1
        self.interaction_nnp_params["approx_channels"] = 1

        le_interaction_nnp_params = self.interaction_nnp_params.copy()
        le_interaction_nnp_params["approx_channels"] = 0
        self.ele_nnp = LocalExcitedStateNNP(
            device = self.device,
            monomer_nnp_params=self.monomer_nnp_params,
            interaction_nnp_params=le_interaction_nnp_params,
            dimer_aev=dimer_aev,
            modified_aev=modified_aev,
            use_belonging=use_belonging,
            output_1e1g=self.return_monomer_terms,
        )
        self.ect_nnp = ChargeTransferStateNNP(
            device = self.device,
            monomer_nnp_params=self.monomer_nnp_params,
            interaction_nnp_params=self.interaction_nnp_params,
            dimer_aev=dimer_aev,
            modified_aev=modified_aev,
            use_belonging=use_belonging,
            output_ipea=self.return_monomer_terms,
        )
        self.vle_nnp = CouplingNNP(
            device = self.device,
            model_params=self.interaction_nnp_params,
            dimer_aev=dimer_aev,
            modified_aev=modified_aev,
            use_belonging=use_belonging,
        )
        self.vte_nnp = CouplingNNP(
            device = self.device,
            model_params=self.interaction_nnp_params,
            dimer_aev=dimer_aev,
            modified_aev=modified_aev,
            use_belonging=use_belonging,
        )
        self.vth_nnp = CouplingNNP(
            device = self.device,
            model_params=self.interaction_nnp_params,
            dimer_aev=dimer_aev,
            modified_aev=modified_aev,
            use_belonging=use_belonging,
        )

    def init_params(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.ele_nnp.init_params(seed)
        self.ect_nnp.init_params(seed)
        self.vle_nnp.init_params(seed)
        self.vte_nnp.init_params(seed)
        self.vth_nnp.init_params(seed)

    def forward(self, species:torch.Tensor, coordinates:torch.Tensor, belongings:torch.Tensor, approx:torch.Tensor = None):
        vct_approx = approx[...,0]
        vle_approx = approx[...,1]
        vte_approx = approx[...,2]
        vth_approx = approx[...,3]

        if self.return_monomer_terms:
            elg, ele = self.ele_nnp(species, coordinates, belongings)
            eip, eea, ect = self.ect_nnp(species, coordinates, belongings, vct_approx)
            vle = self.vle_nnp(species, coordinates, belongings, vle_approx)
            vte = self.vte_nnp(species, coordinates, belongings, vte_approx)
            vth = self.vth_nnp(species, coordinates, belongings, vth_approx)
            return torch.stack([elg, eip, eea, ele, ect, vle, vte, vth], dim = -1).squeeze()
        else:
            ele = self.ele_nnp(species, coordinates, belongings)
            ect = self.ect_nnp(species, coordinates, belongings, vct_approx)
            vle = self.vle_nnp(species, coordinates, belongings, vle_approx)
            vte = self.vte_nnp(species, coordinates, belongings, vte_approx)
            vth = self.vth_nnp(species, coordinates, belongings, vth_approx)
            return torch.stack([ele, ect, vle, vte, vth], dim = -1).squeeze()
