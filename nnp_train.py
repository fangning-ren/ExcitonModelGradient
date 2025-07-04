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

def minimum_loss_sign(target:torch.Tensor, predit:torch.Tensor):
    return torch.abs(predit) * torch.sign(target)

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
    monomername = "perylene"
    dataset_prefix = "PrDim"

    datasetfolder = "data/datasets-ani"
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
    # checkpoint_path = "/home/fren5/asphaltene-ml/trimer-exmod/ml-nn7/models/multitask-nnp-400.pt"
    checkpoint_path = "thisfiledoesnotexist.pt"

    s = sys.argv[1] if len(sys.argv) >= 2 else "ooo"
    use_belonging   = True  # False if s[0] == "o" else True 
    dimer_aev       = False # False if s[1] == "o" else True
    modified_aev    = False # False if s[2] == "o" else True
    s1 = "u" if use_belonging else "o"
    s2 = "d" if dimer_aev     else "o"
    s3 = "m" if modified_aev  else "o"
    suffix = f"{s1}{s2}{s3}"
    jobname = f"{monomername}-{suffix}"
    
    # fix the random seed to make the result reproducible
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size      = 64
    nepoch          = 501
    save_interval   = 100
    learning_rate   = 1.0e-3

    training, validation = create_datasets(dspaths, target_property, approx_property, species_order, align_coupling_sign = False)
    training_loader   = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=1)

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
    model.init_params(seed)
    model = model.to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"checkpoint loaded from {checkpoint_path}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # use a exponential decay learning rate scheduler that decays lr from 1.0e-4 to 1.0e-6 in 500 epochs
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1.0e-6 / 1.0e-3) ** (1 / nepoch), last_epoch=-1)


    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    writer = torch.utils.tensorboard.SummaryWriter("runs/" + jobname)

    mse_sum = torch.nn.MSELoss(reduction='sum')
    

    for epoch in range(nepoch):
        model.train()
        training_term_loss = OrderedDict({t: 0.0 for t in target_property})
        n_train = len(training)
        training_loader_ = training_loader if device != "cpu" else tqdm.tqdm(training_loader)
        # training_loader_ = tqdm.tqdm(training_loader)
        for batch_idx, batch in enumerate(training_loader_):
            species     = batch["species"]    .to(device)
            coordinates = batch["coordinates"].to(device)
            belongings  = batch["belongings"] .to(device)
            approx      = batch["approxs"]    .to(device)
            target      = batch["targets"]    .to(device)
            predit = model(species, coordinates, belongings, approx)
            loss = compute_losses(target, predit, training_term_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_term_loss = {k: (v/n_train)**0.5 for k, v in training_term_loss.items()}
        # plot all loss in a single plot
        writer.add_scalars("training_loss", training_term_loss, epoch)

        total_loss = sum(training_term_loss.values()) / len(training_term_loss)
        print(f"epoch {epoch}: training loss: {total_loss}", flush=True)

        # validation
        model.eval()
        validation_term_loss = OrderedDict({t: 0.0 for t in target_property})
        n_valid = len(validation)
        validation_loader_ = validation_loader if device != "cpu" else tqdm.tqdm(validation_loader)

        predits = []
        targets = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(validation_loader_):
                species     = batch["species"]    .to(device)
                coordinates = batch["coordinates"].to(device)
                belongings  = batch["belongings"] .to(device)
                approx      = batch["approxs"]    .to(device)
                target      = batch["targets"]    .to(device)
                predit = model(species, coordinates, belongings, approx)
                loss = compute_losses(target, predit, validation_term_loss)
                predits.append(predit.cpu())
                targets.append(target.cpu())

        validation_term_loss = {k: (v/n_valid)**0.5 for k, v in validation_term_loss.items()}
        writer.add_scalars("validation_loss", validation_term_loss, epoch)

        predits = torch.cat(predits, dim = 0)
        targets = torch.cat(targets, dim = 0)
        meannss = torch.ones_like(targets) * targets.mean(axis = 0, keepdims = True)
        validation_term_R2 = OrderedDict(
            {k: 1 - mse_sum(predits[:, i], targets[:, i]) / mse_sum(meannss[:, i], targets[:, i]) for i, k in enumerate(validation_term_loss.keys())}
        )
        writer.add_scalars("validation_R2", validation_term_R2, epoch)

        total_loss = sum(validation_term_loss.values()) / len(validation_term_loss)
        print(f"epoch {epoch}: validation loss: {total_loss}", flush=True)
        scheduler.step()
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        if scheduler.state_dict()["_last_lr"][0] < 1e-6:
            break
        
        if epoch % save_interval == 0:
            plot_image(targets.cpu().numpy(), predits.cpu().numpy(), target_property, f"images/{jobname}-{epoch}.png")
            torch.save(model.state_dict(), f"models/{jobname}-{epoch}.pt")






