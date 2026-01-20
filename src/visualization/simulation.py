import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional


def plot_cmb_map(
    map_data: torch.Tensor, 
    c_min: float = -400, 
    c_max: float = 400, 
    x_width: Optional[float] = None, 
    y_width: Optional[float] = None
) -> plt.Figure:
    map_np = map_data.numpy() if not isinstance(map_data, np.ndarray) else map_data
    print("map mean:", np.mean(map_np), "map rms:", np.std(map_np))

    if x_width is None:
        x_width = (2**10) * 0.5 / 60
    if y_width is None:
        y_width = (2**10) * 0.5 / 60

    plt.gcf().set_size_inches(x_width, y_width)
    im = plt.imshow(map_np, interpolation='bilinear', origin='lower', cmap=cm.RdBu_r)
    im.set_clim(c_min, c_max)
    im.set_extent([0, x_width, 0, y_width])
    plt.ylabel(r"Angle $[^\circ]$")
    plt.xlabel(r"Angle $[^\circ]$")

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label="Temperature [$\mu $K]")

    fig = plt.gcf()
    return fig
