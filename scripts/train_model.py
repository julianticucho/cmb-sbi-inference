import os
import torch
import matplotlib.pyplot as plt
from astropy.io import fits
from src.config import PATHS
from src.plotter import Plotter
from src.processor import Processor
from src.trainer import Trainer

def main():
    processor = Processor(type_str="TT")
    theta, x = processor.load_simulations("01_TT_high_ell_unbinned_noise_planck_50000.pt")
    print(theta.shape, x.shape)
    
    inf = Trainer("NPSE", embedding_type="CNN")
    print(inf.model_params)
    _, fig, axes = inf.train(theta, x, plot=True)
    inf.save("NPSE+CNN_01_TT_high_ell_unbinned_noise_planck_50000.pth")
    plt.savefig(os.path.join(PATHS["summary"], "NPSE+CNN_01_TT_high_ell_unbinned_noise_planck_50000.pdf"), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()