import torch

def find_nearest(array, value):
    """Find the index of the element in a tensor that is closest to a given value"""
    idx = (torch.abs(array - value)).argmin()
    return idx

def compute_bins(lmin, lmax, Nbins):
    """Calculates bins using PyTorch tensors"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ls = torch.arange(lmax+1, device=device)
    num_modes = torch.zeros(lmax+1, device=device)
    cumulative_num_modes = torch.zeros(lmax+1, device=device)
    bin_edges = torch.zeros(Nbins+1, dtype=torch.long, device=device)
    bin_edges[0] = lmin
    cumulative = 0

    for i in range(lmin, lmax+1):
        num_modes[i] = 2*i + 1
        cumulative += num_modes[i]
        cumulative_num_modes[i] = cumulative

    Num_modes_total = num_modes.sum()
    print("Total number of modes in (l_min,l_max) = ", Num_modes_total.item())
    Num_modes_per_bin = Num_modes_total / Nbins
    print("Number of modes in each bin = ", Num_modes_per_bin.item())

    for i in range(1, Nbins+1):
        target = Num_modes_per_bin * i
        bin_edges[i] = find_nearest(cumulative_num_modes, target)
    
    return Num_modes_per_bin, cumulative_num_modes, bin_edges

def bin_power(ell, cl, bins):
    """Binning power spectrum using PyTorch tensors"""
    device = cl.device
    bins = bins.to(device)
    ell = ell.to(device)
    sorted_bins, _ = torch.sort(bins)
    bin_indices = torch.bucketize(ell, sorted_bins, right=False)
    
    count = torch.zeros(len(bins)-1, device=device)
    cl_bin_sum = torch.zeros(len(bins)-1, device=device)
    el_med_sum = torch.zeros(len(bins)-1, device=device)
    
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        count[i-1] = mask.sum()
        cl_bin_sum[i-1] = cl[mask].sum()
        el_med_sum[i-1] = (ell[mask] * cl[mask]).sum()
    
    nonzero_mask = cl_bin_sum != 0
    el_med = torch.zeros_like(el_med_sum, dtype=torch.long)
    el_med[nonzero_mask] = (el_med_sum[nonzero_mask] / cl_bin_sum[nonzero_mask]).long()
    
    cl_bin = torch.zeros_like(cl_bin_sum)
    cl_bin[count != 0] = cl_bin_sum[count != 0] / count[count != 0]
    
    return el_med, cl_bin, count

def bin_simulations(x, lmin, lmax, Nbins):
    """Binning simulations using PyTorch tensors"""
    device = x.device
    _, _, bin_edges = compute_bins(lmin, lmax, Nbins)
    bin_edges = bin_edges.to(device)
    ell = torch.arange(lmax+1, device=device)
    num_sims = x.shape[0]
    
    x_binned = torch.zeros((num_sims, Nbins), device=device)
    l_binned = torch.zeros((num_sims, Nbins), dtype=torch.long, device=device)

    for i in range(num_sims):
        l_med, cl_bin, _ = bin_power(ell, x[i], bin_edges)
        l_binned[i] = l_med
        x_binned[i] = cl_bin

    return l_binned, x_binned