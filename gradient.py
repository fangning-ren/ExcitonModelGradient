import os
import sys
import time
from collections import OrderedDict

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.tensorboard

import torchani
from torchani.aev import AEVComputer


from nnp_dataset import *
from nnp_models import *

def compute_numerical_gradient(model, species, coordinates, belongings, approx, epsilon=1.0e-4):
    # Ensure coordinates is a torch tensor
    
    coordinates = coordinates.clone().detach().requires_grad_(False)
    #print("Coordinates shape:", coordinates.shape)
    #coordinates = coordinates[0] # squeeze the batch_size
    #print("Coordinates shape2 :", coordinates.shape)
    
    batch_size, N_atoms, dim = coordinates.shape
    #N_atoms, dim = coordinates.shape
    #print("Coordinates shape3:", coordinates.shape)
    #print("The number of atoms:")
    #print(N_atoms)
    #print("Dimesnions")
    #print(dim)
    
    print("\n\n")
    grad = torch.zeros_like(coordinates)
    print("Espilon value: ", epsilon)
    # Loop over each coordinate component
    for b in range(batch_size):
        for i in range(N_atoms):
            for j in range(dim):
                # Perturb in + direction
                coords_plus = coordinates.clone()
                coords_plus[b, i, j] += epsilon

                mask = [True, False, False, False,False, False, False, False]
                # From a list
                tensor_from_list = torch.tensor(mask)
                #print(f"Tensor from list: {mask}")
                E_plus = model(species, coords_plus, belongings, approx)
                E_plus_1e1g = torch.masked_select(E_plus, tensor_from_list)

                # Perturb in - direction
                coords_minus = coordinates.clone()
                coords_minus[b, i, j] -= epsilon
                E_minus = model(species, coords_minus, belongings, approx)
                E_minus_1e1g = torch.masked_select(E_minus, tensor_from_list)
                # Central difference
                #print("The gradient of", b, "st batch_size: ")
                grad[b, i, j] = (E_plus_1e1g - E_minus_1e1g) / (2 * epsilon)
    print("The numerical gradient of the above coordinates are given by: ")
    print(grad)
    return grad

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Typical procedure to save & load model across devices
    monomername = "perylene"
    dataset_prefix = "PrDim"
    use_belonging   = True  # False if s[0] == "o" else True 
    dimer_aev       = False # False if s[1] == "o" else True
    modified_aev    = False # False if s[2] == "o" else True

    # fix the random seed to make the result reproducible
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size      = 1
    nepoch          = 1
    save_interval   = 100
    learning_rate   = 1.0e-3

    monomer_nnp_params = {
        "elements": ["H", "C",],
        "hidden_channels": [256, 256, 256],
        "output_channels": 3,   # monomer properties
        "approx_channels": 0,
        "Rcr": 7.5000e+00, "EtaR": 1.6000000e+01, 
        "ShfR_min": 0.9, "ShfR_n": 16,
        "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
        "ShfA_min": 0.9, "ShfA_n": 4 ,
        "Zeta": 3.2000000e+01, "ShfZ_n": 8,
    }
    interaction_nnp_params = {
        "elements": ["H", "C",],
        "hidden_channels": [64, 64, 64],
        "output_channels": 5,   # interaction properties
        "approx_channels": 4,   # approximations
        "Rcr": 7.5000e+00, "EtaR": 1.6000000e+01, 
        "ShfR_min": 0.9, "ShfR_n": 16,
        "Rca": 3.5000e+00, "EtaA": 8.0000000e+00,
        "ShfA_min": 0.9, "ShfA_n": 4 ,
        "Zeta": 3.2000000e+01, "ShfZ_n": 8,
    }
    model = ExmodSeparateNNP(
        device   = device,
        monomer_nnp_params     = monomer_nnp_params,
        interaction_nnp_params = interaction_nnp_params,
        use_belonging          = use_belonging,
        dimer_aev              = dimer_aev,
        modified_aev           = modified_aev,
        return_monomer_terms   = True,
    )

    checkpoint_path = "thisfiledoesnotexist.pt"
    datasetfolder = "data/datasets-ani"
    model.init_params(seed)
    model = model.to(device)
    #print(model)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"checkpoint loaded from {checkpoint_path}", flush=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # use a exponential decay learning rate scheduler that decays lr from 1.0e-4 to 1.0e-6 in 500 epochs
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1.0e-6 / 1.0e-3) ** (1 / nepoch), last_epoch=-1)
    print("Gradient calculation attempt starts here: ")

    modelfilepery = "models/perylene-best.pt"
    # already defined : device = torch.device('cpu') 
    model.load_state_dict(torch.load(modelfilepery, map_location=device))
    # print("Gradient calculation: STEP 1: LOAD FILE using print does not work")
    #print(model)

    ''' 
    print("All model attributes:")
    for attr in dir(model):
        if not attr.startswith("_"):
            print(attr)
            #model.state_dict
    '''

    #print("Gradient calculation: STEP 2: LOAD FILE using eval")
    #model.eval()

    for name, param in model.named_parameters():
        #if param.requires_grad:
        # print(name, param.shape)
        pass

    # Attempting to print only monomeric elements: 
    # print("Attempt to print only the monomer elements")
    # print(model.ele_nnp.monomer_model.element_network[:][:])
    
    first_linear = model.ele_nnp.monomer_model.element_network[0]
    # print(first_linear.weight[:, 0])

    # Use param to print elements in model
    print("Gradient calculation: STEP 3: LOAD FILE using eval")
    #print("Model Parameters")
    #tensor = dict(model.named_parameters())[0]  # replace with actual name
    #print(tensor[:, 0])  # prints the first column
    
    # 64 parameters tensors are avaiable: I believe this is 8*8
    for param in model.parameters():
        #print(param)
        pass

    save_interval   = 100
    learning_rate   = 1.0e-3

    dspaths = [
        os.path.join(datasetfolder, f"ani-{dataset_prefix}-COM4A.h5"),
        os.path.join(datasetfolder, f"ani-{dataset_prefix}-NST5A.h5"),
        os.path.join(datasetfolder, f"ani-{dataset_prefix}-SEP5A.h5"),
    ]
    target_property = [
        "1e(1)g",  "1ip", "2ea",                # monomeric properties
        "1e(1)", "1+2-",                        # diagonal terms
        "1e(1)2e(1)", "1e(1)1+2-", "1e(1)1-2+"  # off-diagonal terms
    ]
    approx_property = [
        "atmapx_v_1+2-",        # coulomb interaction between monomers at 1+2- charge transfer state
        "atmapx_1e(1)2e(1)",    # transition density interaction between the two local excited states
        "atmapx_1e(1)1+2-",     # overlap integral between the two monomers (lumo)
        "atmapx_1e(1)1-2+"      # overlap integral between the two monomers (hmmo) # This is not the correct abbreviation
        ]
    
    species_order = ["H", "C",]

    training, validation = create_datasets(dspaths, target_property, approx_property, species_order, align_coupling_sign = False)
    training_loader   = DataLoader(training, batch_size=batch_size, shuffle=False, num_workers=1)
    training_loader_ = training_loader if device != "cpu" else tqdm.tqdm(training_loader)

    predits = []
    targets = []
    
    
    print("\n\nBelow, the coordinates are printed followed with numerical and autodiff graidents \n\n")

    for batch_idx, batch in enumerate(training_loader_):
            training_term_loss = OrderedDict({t: 0.0 for t in target_property})
            coordinates = batch["coordinates"].to(device).to(device).float().requires_grad_(True)
            print("Coordinates for batch_idx: ", batch_idx, "\n\n")
            print(coordinates)

            # There are 600 coordinates tensors
            # model.backward(): This does not work ..you need to compute on a scalar quantity.
            # print("Attempt to compute gradients: ")
            # print(coordinates.grad)
            species     = batch["species"]    .to(device)
            belongings  = batch["belongings"] .to(device)
            approx      = batch["approxs"]    .to(device)
            target      = batch["targets"]    .to(device)

            #print("Coordinates shape:", coordinates.shape)
            #N_atoms, dim = coordinates.shape
            #print("Coordinates shape:", coordinates.shape)
            #print("The number of atoms:")
            #print(N_atoms)
            #print("Dimesnions")
            #print(dim)
            # Code to compute numerical gradients: 
            
            # Compute and Print the numerical gradients
            #print("batch: ", batch)
            epsilon = 1.0e-4
            gradients = compute_numerical_gradient(model, species, coordinates, belongings, approx, epsilon)
            #print(gradients)
            


            output = model(species, coordinates, belongings, approx)
            #loss = compute_losses(target, output, training_term_loss)
            #print(loss)
            # It had computed 600 losses. 
            #print("The output of model: ")
            #print(output)
            mask = [True, False, False, False,False, False, False, False]
            # From a list
            tensor_from_list = torch.tensor(mask)
            #print(f"Tensor from list: {mask}")
            loss = torch.masked_select(output, tensor_from_list)
            #print(loss)
            #output.backward() only on a scalar quantity
            loss.backward()
            print("The following are the gradients obtained USING AUTODIFF:")
            print(coordinates.grad)
            
            #print("END OF GRADIENTS!")
            #plot_image(loss.detach().numpy(), coordinates.grad.numpy(), target_property, f"images/{batch}-perylene-best.png")
            print("\n\n\n\n\n")
    #plot_image(targets.cpu().numpy(), output.cpu().numpy(), target_property, f"images/gradients.png")
    
       






