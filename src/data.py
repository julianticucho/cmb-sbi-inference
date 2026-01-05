import torch
import os
from astropy.io import fits
from src.config import PATHS

class Dataset:
    def __init__(self):
        pass
    
    @staticmethod
    def import_data():
        path = os.path.join(PATHS["planck"],"COM_PowerSpect_CMB_R1.10.fits")
        hdul = fits.open(path)

        low_data = hdul[1].data
        high_data = hdul[2].data
        cov_matrix = hdul[3].data

        ell_low = low_data["ELL"]
        dell_low = low_data["D_ELL"]
        err_low = [low_data["ERRDOWN"], low_data["ERRUP"]]
        ell_high = high_data["ELL"]
        dell_high = high_data["D_ELL"]
        lmin = high_data["LMIN"]
        lmax = high_data["LMAX"]
        err_high = high_data["ERR"]

        ell_low = ell_low.astype(ell_low.dtype.newbyteorder('='))
        dell_low = dell_low.astype(dell_low.dtype.newbyteorder('='))
        err_low = [err_low[0].astype(err_low[0].dtype.newbyteorder('=')), err_low[1].astype(err_low[1].dtype.newbyteorder('='))]
        ell_high = ell_high.astype(ell_high.dtype.newbyteorder('='))
        dell_high = dell_high.astype(dell_high.dtype.newbyteorder('='))
        lmin = lmin.astype(lmin.dtype.newbyteorder('='))
        lmax = lmax.astype(lmax.dtype.newbyteorder('='))
        err_high = err_high.astype(err_high.dtype.newbyteorder('='))
        cov_matrix = cov_matrix.astype(cov_matrix.dtype.newbyteorder('='))

        ell_low = torch.from_numpy(ell_low)
        dell_low = torch.from_numpy(dell_low)
        err_low = [torch.from_numpy(err_low[0]), torch.from_numpy(err_low[1])]
        ell_high = torch.from_numpy(ell_high)
        dell_high = torch.from_numpy(dell_high)
        lmin = torch.from_numpy(lmin)
        lmax = torch.from_numpy(lmax)
        err_high = torch.from_numpy(err_high)
        cov_matrix = torch.from_numpy(cov_matrix)

        return ell_low, dell_low, err_low, ell_high, dell_high, lmin, lmax, err_high, cov_matrix
    
