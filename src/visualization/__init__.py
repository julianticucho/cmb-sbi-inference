import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'high-vis'])
plt.rcParams['figure.dpi'] = 300

from .diagnostic import (
    plot_ppc, plot_hpd, plot_sbc, plot_tarp, 
    plot_hpd_tarp_diagnostics, plot_hpd_legacy,
    plot_data_ppc
)
from .inference import correlation_matrix
from .simulation import plot_cmb_map, plot_cl
from . import api

__all__ = [
    'plot_ppc',
    'plot_hpd', 
    'plot_sbc',
    'plot_tarp',
    'plot_hpd_tarp_diagnostics',
    'plot_hpd_legacy',
    'plot_data_ppc',
    'correlation_matrix',
    'plot_cmb_map',
    'plot_cl',
    'api'
]
