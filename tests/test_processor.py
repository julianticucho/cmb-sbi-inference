import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from srcOOP.processor import Processor
from srcOOP.generator import Generator
from srcOOP.config import PARAMS

def test_instantiation():
    processor = Processor(type_str="TT", K=5, device="cpu")
    assert processor.type_str == "TT"
    assert processor.K == 5
    assert processor.device == "cpu"

def test_add_instrumental_noise():
    gen = Generator(type_str="TT", num_workers=1, seed=1)
    _, x = gen.generate_cosmologies(num_simulations=2)
    assert x.shape == torch.Size([2, 2551])
    processor = Processor(type_str="TT")
    x_noisy = processor.add_instrumental_noise(x)
    assert x_noisy.shape == torch.Size([2, 2551])

def test_sample_observed_spectra():
    gen = Generator(type_str="TT", num_workers=1, seed=1)
    _, x = gen.generate_cosmologies(num_simulations=2)
    processor = Processor()
    x_obs = processor.sample_observed_spectra(x)
    assert x_obs.shape == torch.Size([2, 2551])
    # probability of all values being the same is almost 0
    assert not torch.allclose(x, x_obs)

def test_bin_spectra_reduces_length():
    gen = Generator(type_str="TT", num_workers=1, seed=1)
    _, x = gen.generate_cosmologies(num_simulations=1)
    processor = Processor(type_str="TT")
    x_binned = processor.bin_spectra(x, bin_width=100)
    # must to return a vector of length 100
    expected_len = 100
    assert x_binned.shape[1] == expected_len

def test_generate_noise_multiple_shapes():
    gen = Generator(type_str="TT", num_workers=1, seed=1)
    theta, x = gen.generate_cosmologies(num_simulations=2)
    processor = Processor(K=3)
    theta_expanded, x_noisy = processor.generate_noise_multiple(x, theta)
    # must to triplicate theta and x and return tensors of shape (6, 5) and (6, 2551)
    assert theta_expanded.shape == torch.Size([6, 5])
    assert x_noisy.shape == torch.Size([6, 2551])
    assert not torch.allclose(x_noisy[:2], x_noisy[2:4])

def test_create_simulator():
    theta = [0.022, 0.12, 1.041, 3.0, 0.965]
    processor = Processor(type_str="TT", K=1)
    simulator = processor.create_simulator(noise=True, binning=False)
    assert simulator is not None
    assert callable(simulator)
    print(simulator([0.022, 0.12, 1.041, 3.0, 0.965])) 

def test_select_component():
    gen = Generator(type_str="TT+EE+BB+TE", num_workers=1, seed=1)
    _, x = gen.generate_cosmologies(num_simulations=1)
    processor = Processor(type_str="TT+EE+BB+TE")
    x_selected = processor.select_components(x, TT=True, EE=True)
    assert x_selected.shape == torch.Size([1, 5102])


 
