import os
import torch
import matplotlib.pyplot as plt
from src.plotter import Plotter
from src.processor import Processor
from src.trainer import Trainer
from src.config import PATHS

experiments = [
    {
        "model_name": "NPSE",
        "simulation_file": "01_all_Cls_reduced_prior_50000.pt",
        "posterior_file": "01_NPSE_TT_reduced_prior_50000.pth",
        "type_str": "TT",
        "TT": True,
        "EE": False,
        "BB": False,
        "TE": False,
        "noise": False,
        "binning": False,
        "color": "#550A41",
        "filled": True,
        "label": "Seed 1",
        "num_samples": 25000
    },
    {
        "model_name": "NPSE",
        "simulation_file": "02_all_Cls_reduced_prior_50000.pt",
        "posterior_file": "02_NPSE_TT_reduced_prior_50000.pth",
        "type_str": "TT",
        "TT": True,
        "EE": False,
        "BB": False,
        "TE": False,
        "noise": False,
        "binning": False,
        "color": "#000000",
        "filled": False,
        "label": "Seed 2",
        "num_samples": 25000
    },
    {
        "model_name": "NPSE",
        "simulation_file": "03_all_Cls_reduced_prior_50000.pt",
        "posterior_file": "03_NPSE_TT_reduced_prior_50000.pth",
        "type_str": "TT",
        "TT": True,
        "EE": False,
        "BB": False,
        "TE": False,
        "noise": False,
        "binning": False,
        "color": "#006FED",
        "filled": True,
        "label": "Seed 3",
        "num_samples": 25000
    }
]

def generate_samples(
    model_name: str,
    simulation_file: str,
    posterior_file: str,
    type_str: str,
    TT: bool,
    EE: bool,
    BB: bool,
    TE: bool,
    noise: bool,
    binning: bool,
    num_samples: int,
    true_parameter: list,
):
    processor = Processor(type_str=type_str)
    trainer = Trainer(model_name)
    theta, x = processor.load_simulations(simulation_file)
    x = processor.select_components(x, TT=TT, EE=EE, BB=BB, TE=TE)
    trainer.load_posterior(posterior_file, theta, x)
    samples = trainer.sample(type_str=type_str, true_parameter=true_parameter, 
                             num_samples=num_samples, noise=noise, binning=binning)
    return samples

true_parameter1 = [0.02212, 0.1206, 1.04077, 3.04, 0.9626]
true_parameter2 = [0.02205, 0.1224, 1.04035, 3.028, 0.9589]
true_parameter3 = [0.02218, 0.1198, 1.04052, 3.052, 0.9672]

samples_list = []
colors = []
labels = []
filled_flags = []

for exp in experiments:
    samples = generate_samples(
        model_name=exp["model_name"],
        simulation_file=exp["simulation_file"],
        posterior_file=exp["posterior_file"],
        type_str=exp["type_str"],
        TT=exp["TT"],
        EE=exp["EE"],
        BB=exp["BB"],
        TE=exp["TE"],
        noise=exp["noise"],
        binning=exp["binning"],
        num_samples=exp["num_samples"],
        true_parameter=true_parameter1,
    )
    samples_list.append(samples)
    colors.append(exp["color"])
    labels.append(exp["label"])
    filled_flags.append(exp["filled"])

plotter = Plotter.from_config()
fig = plotter.plot_confidence_contours(
    samples_list,
    true_parameter1,  
    sample_labels=labels,
    sample_colors=colors,
    filled=filled_flags
)

plt.savefig(os.path.join(PATHS["confidence"], "testt.pdf"), bbox_inches='tight')
plt.close('all')
del fig

# exp = experiments[0]

# processor = Processor(type_str=exp["type_str"])
# trainer = Trainer(exp["model_name"])
# plotter = Plotter.from_config()

# theta, x = processor.load_simulations(exp["simulation_file"])
# x = processor.select_components(x, TT=exp["TT"], EE=exp["EE"], BB=exp["BB"], TE=exp["TE"])
# trainer.load_posterior(exp["posterior_file"], theta, x)
# posterior = trainer.posterior  # <- distribución entrenada

# fig, coverage = plotter.plot_posterior_calibration(
#     posterior=posterior,
#     simulator=processor.create_simulator(),  
#     num_posterior_samples=2000,
#     num_true_samples=1000,         
#     device="cpu"
# )

# plt.savefig(os.path.join(PATHS["calibration"], "testt.pdf"), bbox_inches='tight')
# plt.close(fig)
