from src.processor import Processor
from src.data import Dataset

def unbox_data():
    dataset = Dataset()
    _, _, _, _, _, lmin, lmax, _, cov_matrix = dataset.import_data()
    return lmin, lmax, cov_matrix

def simulator(theta):
    processor = Processor(type_str="TT")
    lmin, lmax, cov_matrix = unbox_data()

    simulator_fn = processor.create_simulator()
    x = simulator_fn(theta)
    x = x[30:2478]
    x = processor.bin_high_ell(x, lmin, lmax)
    return x