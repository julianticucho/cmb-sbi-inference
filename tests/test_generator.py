import pytest
import torch
import numpy as np
from srcOOP.generator import Generator  

def test_instantiation():
    gen = Generator()
    assert gen.type_str == "TT+EE+BB+TE"
    assert gen.num_workers == 11
    assert gen.seed == 1

def test_compute_spectrum_shape():
    theta = [0.022, 0.12, 1.041, 3.0, 0.965]  # example parameters
    gen = Generator()
    spec = gen.compute_spectrum(theta)
    # must to return a vector of length 4*2551=10204
    assert spec.ndim == 1
    assert spec.shape[0] == 10204
    
def test_split_spectra():
    theta = [0.022, 0.12, 1.041, 3.0, 0.965]
    gen = Generator()
    spec = gen.compute_spectrum(theta)  
    comps = gen.split_spectra(spec)
    assert set(comps.keys()) == {"TT", "EE", "BB", "TE"}
    assert len(comps["TT"]) == 2551
    assert len(comps["TE"]) == 2551
    assert len(comps["EE"]) == 2551
    assert len(comps["BB"]) == 2551

def test_create_simulator_output():
    """
    Test that the simulator function created by create_simulator() returns a
    tensor with the correct shape.
    """
    theta = [0.022, 0.12, 1.041, 3.0, 0.965]

    gen_tt = Generator(type_str="TT")
    simulator = gen_tt.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2551

    gen_ee = Generator(type_str="EE")
    simulator = gen_ee.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2551

    gen_bb = Generator(type_str="BB")
    simulator = gen_bb.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2551

    gen_te = Generator(type_str="TE")
    simulator = gen_te.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2551

    gen_tt_ee = Generator(type_str="TT+EE")
    simulator = gen_tt_ee.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551*2 = 5102
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 5102

    gen_bb_te = Generator(type_str="BB+TE")
    simulator = gen_bb_te.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551*2 = 5102
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 5102

    gen_ee_te = Generator(type_str="EE+TE")
    simulator = gen_ee_te.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551*2 = 5102
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 5102

    gen_tt_ee_te = Generator(type_str="TT+EE+TE")
    simulator = gen_tt_ee_te.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551*3 = 7653
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 7653

    gen_tt_ee_bb_te = Generator(type_str="TT+EE+BB+TE")
    simulator = gen_tt_ee_bb_te.create_simulator()
    out = simulator(theta)
    # must to return a vector of length 2551*4 = 10204
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 10204

# note: test of generate_cosmologies() can take a long time if num_simulations is large
def test_generate_cosmologies_small():
    gen = Generator(num_workers=1)
    theta, x = gen.generate_cosmologies(num_simulations=2)
    assert isinstance(theta, torch.Tensor)
    assert isinstance(x, torch.Tensor)
    assert theta.shape[0] == 2
    assert x.shape[0] == 2
    assert x.shape[1] == 10204
    assert theta.shape[1] == 5