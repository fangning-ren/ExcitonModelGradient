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

# Use FR's loss function to compute loss and evaluate gradients wrt coordinates
def compute_losses(
        target:torch.Tensor, 
        predit:torch.Tensor,
        current_term_loss:OrderedDict,
    ):
    losses = (predit - target) ** 2
    for i, key in enumerate(current_term_loss.keys()):
        current_term_loss[key] += losses[:, i].sum().item()
    total_loss = (losses.sum() / (target.shape[0] * target.shape[1])) ** 0.5    # RMSE
    return total_loss

def plot_image(target, predict, terms, filename):
    fig, axs = plt.subplots(3, 3, figsize = (10, 10), dpi = 200)
    axs = axs.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for i, (ax, t, p) in enumerate(zip(axs, target.T, predict.T)):
        ax.scatter(t, p, s = 1, alpha = 0.1)
        ax.plot([t.min(), t.max()], [t.min(), t.max()], color = "black", linestyle = "--")
        ax.text(0.05, 0.95, terms[i], transform=ax.transAxes, fontsize=8, verticalalignment='top')
        ax.tick_params(direction='in')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    plt.clf()

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

    batch_size      = 64
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
        print(name, param.shape)

    # Attempting to print only monomeric elements: 
    print("Attempt to print only the monomer elements")
    print(model.ele_nnp.monomer_model.element_network[:][:])
    
    first_linear = model.ele_nnp.monomer_model.element_network[0]
    print(first_linear.weight[:, 0])

    # Use param to print elements in model
    print("Gradient calculation: STEP 3: LOAD FILE using eval")
    #print("Model Parameters")
    #tensor = dict(model.named_parameters())[0]  # replace with actual name
    #print(tensor[:, 0])  # prints the first column
    
    # 64 parameters tensors are avaiable: I believe this is 8*8
    for param in model.parameters():
        #print(param)
        zz = 1
        
    batch_size = 64
    nepoch          = 3
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
    training_loader   = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=1)
    training_loader_ = training_loader if device != "cpu" else tqdm.tqdm(training_loader)

    predits = []
    targets = []
    
    print("The coordinates are being printed")
    for batch_idx, batch in enumerate(training_loader_):
            training_term_loss = OrderedDict({t: 0.0 for t in target_property})
            coordinates = batch["coordinates"].to(device).to(device).float().requires_grad_(True)
            print(coordinates)

            # There are 600 coordinates tensors
            # model.backward(): This does not work ..you need to compute on a scalar quantity.
            # print("Attempt to compute gradients: ")
            # print(coordinates.grad)
            species     = batch["species"]    .to(device)
            belongings  = batch["belongings"] .to(device)
            approx      = batch["approxs"]    .to(device)
            target      = batch["targets"]    .to(device)
            output = model(species, coordinates, belongings, approx)
            loss = compute_losses(target, output, training_term_loss)
            #print(loss)
            
            # It had computed 600 losses. 
            # print(output)
            #output.backward() only on a scalar quantity
            
            loss.backward()
            
            print("The following are the gradients: obtained from model using perylene-best.pt, coordinates;")
            print("USING AUTODIFF>>>>>")
            
            print(coordinates.grad)
            
            #print("END OF GRADIENTS!")
            #plot_image(loss.detach().numpy(), coordinates.grad.numpy(), target_property, f"images/{batch}-perylene-best.png")

    #plot_image(targets.cpu().numpy(), output.cpu().numpy(), target_property, f"images/gradients.png")
    
       






