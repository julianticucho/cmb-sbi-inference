from src.trainer import Trainer
from src.processor import Processor
from src.config import PATHS

def main():
    processor = Processor(type_str="TT+EE+BB+TE")
    theta, x = processor.load_simulations("03_TT_high_ell_binned_noise_planck_50000.pt")
    print(theta.shape, x.shape)
    
    inf = Trainer("NPSE")
    inf.train(theta, x, plot=False)
    inf.save("03_NPSE_TT_high_ell_binned_noise_planck_50000.pth")

if __name__ == "__main__":
    main()