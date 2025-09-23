import os
import pytest
import torch
from src.generator import Generator
from src.trainer import Trainer
from src.config import PATHS


@pytest.fixture
def dummy_data():
    gen = Generator(type_str="TT", num_workers=1, seed=1)
    theta, x = gen.generate_cosmologies(num_simulations=5)
    return theta, x

@pytest.mark.parametrize("method", ["SNPE_C", "NPSE", "FMPE", "NLE", "MNLE"])
def test_inference_creation(method):
    """Testing inference creation with different methods"""
    inf = Trainer(method=method)
    assert inf.inference is not None

@pytest.mark.parametrize("method", ["SNPE_C", "NPSE", "FMPE", "NLE"]) # MNLE not supported yet
def test_train_and_build_posterior(dummy_data, method):
    """Testing training and building posterior with different methods"""
    theta, x = dummy_data
    inf = Trainer(method=method)
    density_estimator, fig, axes = inf.train(theta, x, plot=False)

    assert density_estimator is not None
    posterior = inf.build_posterior()
    assert posterior is not None

def test_save_and_load(dummy_data):
    """Testing save and load with different methods"""
    theta, x = dummy_data
    inf = Trainer(method="NPSE")
    inf.train(theta, x, plot=False)

    filename = "test_model.pth"
    inf.save(filename)
    assert os.path.exists(os.path.join(PATHS["models"], filename))

    new_inf = Trainer(method="NPSE")
    posterior = new_inf.load(filename, theta, x)
    assert posterior is not None

def test_sample(dummy_data):
    """Testing sample of the posterior"""
    theta, x = dummy_data
    inf = Trainer(method="NPSE")
    inf.train(theta, x, plot=False)
    inf.build_posterior()

    true_parameter = torch.tensor([0.022, 0.12, 1.041, 3.0, 0.965])
    samples = inf.sample(true_parameter, type_str="TT", noise=False, binning=False, num_samples=10)

    assert isinstance(samples, torch.Tensor)
    assert samples.shape[0] == 10

def test_build_posterior_without_training():
    """Testing posterior cannot be built without training"""
    est = Trainer(method="NPSE")
    with pytest.raises(RuntimeError):
        est.build_posterior()

