__all__ = ["calc_kl", "calc_js", "est_pdf", "integrate", "calculate_divergences", "calculate_l1_and_l2_norm_errors",
           "eps"]

import itertools

import numpy as np
import torch

eps = 1e-3


def calc_kl(pk, qk):
    '''
    A function to calculate the Kullback-Libler divergence between p and q distribution.
    Arguments:
    pk: pdf values from p distribution;
    qk: pdf values from q distribution.
    '''
    return np.nan_to_num(pk * np.log(pk / qk))


def integrate(bins, dx=1):
    return np.trapz(bins, dx=dx)


def calc_js(pk, qk):
    '''
    A function to calculate the Jensen-Shanon divergence between p and q distribution.
    Arguments:
    pk: pdf values from p distribution
    qk: pdf values from q distribution
    '''
    mk = 0.5 * (pk + qk)
    return 0.5 * (calc_kl(pk, mk) + calc_kl(qk, mk))


def est_pdf(hist_counts, beta=1):
    '''
    A function to make pdf estimation using a generalization of Laplace rule. Using that we can avoid bins with zero probability
    This implementation is based on:
    https://papers.nips.cc/paper/2001/file/d46e1fcf4c07ce4a69ee07e4134bcef1-Paper.pdf
    Arguments:
    hist_counts: the histogram counts
    beta: the beta factor for the probability estimation
    '''
    K = len(hist_counts)
    kappa = K * beta
    pdf = (hist_counts + beta) / (hist_counts.sum() + kappa)
    return pdf


#
# Calculate kl pair to pair using permutation between real and fake samples
#
def calculate_divergences(real_samples, fake_samples):
    kl = [];
    js = [];
    l1 = []
    for r_idx, f_idx in itertools.permutations(list(range(real_samples.shape[0])), 2):
        # Int_inf_to_plus_int p(x)/(dx*total) * dx = 1
        #real_pdf, bins = np.histogram(real_samples[r_idx].flatten(), bins=100, range=(0, 1), density=True)
        real_pdf, bins = np.histogram(real_samples[r_idx], bins=100, range=(0, 1), density=True)
        #fake_pdf, bins = np.histogram(fake_samples[f_idx].flatten(), bins=100, range=(0, 1), density=True)
        fake_pdf, bins = np.histogram(fake_samples[f_idx], bins=100, range=(0, 1), density=True)
        kl.append(integrate(calc_kl(pk=real_pdf + eps, qk=fake_pdf + eps), dx=1 / 100))
        js.append(integrate(calc_js(pk=real_pdf + eps, qk=fake_pdf + eps), dx=1 / 100))
    return kl, js


#
# Calculate l1/l2 pair to pair using permutation between real and fake samples
#
def calculate_l1_and_l2_norm_errors(real_samples, fake_samples):
    l1 = [];
    l2 = []
    for r_idx, f_idx in itertools.permutations(list(range(real_samples.shape[0])), 2):
        # height x width
        #hw = len(real_samples[r_idx]) * len(real_samples[r_idx][0])
        hw = len(real_samples[r_idx])
        # calculate l1
        #l1.append(sum(sum(abs(real_samples[r_idx] - fake_samples[f_idx])))[0] / hw)
        l1.append(np.sum(np.sum(abs(real_samples[r_idx] - fake_samples[f_idx]))) / hw)
        # calculate l2
        # l2_norm_error = 1/HW * Sum ( (y-yhat)**2 )
        #l2.append(sum(sum(np.power(real_samples[r_idx] - fake_samples[f_idx], 2)))[0] / hw)
        l2.append(np.sum(np.sum(np.power(real_samples[r_idx] - fake_samples[f_idx], 2))) / hw)

    return l1, l2

def calculate_divergences_1d(real_samples, fake_samples):
    
    # Int_inf_to_plus_int p(x)/(dx*total) * dx = 1
    #real_pdf, bins = np.histogram(real_samples[r_idx].flatten(), bins=100, range=(0, 1), density=True)
    real_pdf, bins = np.histogram(real_samples, bins=100, range=(0, 1), density=True)
    #fake_pdf, bins = np.histogram(fake_samples[f_idx].flatten(), bins=100, range=(0, 1), density=True)
    fake_pdf, bins = np.histogram(fake_samples, bins=100, range=(0, 1), density=True)
    kl = integrate(calc_kl(pk=real_pdf + eps, qk=fake_pdf + eps), dx=1 / 100)
    js = integrate(calc_js(pk=real_pdf + eps, qk=fake_pdf + eps), dx=1 / 100)
    return kl, js


def image_hist2d(image: torch.Tensor, min: float = 0., max: float = 255.,
                 n_bins: int = 256, bandwidth: float = -1.,
                 centers: torch.Tensor = torch.tensor([]), return_pdf: bool = False):
    """Function that estimates the histogram of the input image(s).

    The calculation uses triangular kernel density estimation.

    Args:
        x: Input tensor to compute the histogram with shape
        :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        min: Lower end of the interval (inclusive).
        max: Upper end of the interval (inclusive). Ignored when
        :attr:`centers` is specified.
        n_bins: The number of histogram bins. Ignored when
        :attr:`centers` is specified.
        bandwidth: Smoothing factor. If not specified or equal to -1,
        bandwidth = (max - min) / n_bins.
        centers: Centers of the bins with shape :math:`(n_bins,)`.
        If not specified or empty, it is calculated as centers of
        equal width bins of [min, max] range.
        return_pdf: If True, also return probability densities for
        each bin.

    Returns:
        Computed histogram of shape :math:`(bins)`, :math:`(C, bins)`,
        :math:`(B, C, bins)`.
        Computed probability densities of shape :math:`(bins)`, :math:`(C, bins)`,
        :math:`(B, C, bins)`, if return_pdf is ``True``. Tensor of zeros with shape
        of the histogram otherwise.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}.")

    if centers is not None and not isinstance(centers, torch.Tensor):
        raise TypeError(f"Bins' centers type is not a torch.Tensor. Got {type(centers)}.")

    if centers.numel() > 0 and centers.dim() != 1:
        raise ValueError(f"Bins' centers must be a torch.Tensor "
                         "of the shape (n_bins,). Got {values.shape}.")

    if not isinstance(min, float):
        raise TypeError(f'Type of lower end of the range is not a float. Got {type(min)}.')

    if not isinstance(max, float):
        raise TypeError(f"Type of upper end of the range is not a float. Got {type(min)}.")

    if not isinstance(n_bins, int):
        raise TypeError(f"Type of number of bins is not an int. Got {type(n_bins)}.")

    if bandwidth != -1 and not isinstance(bandwidth, float):
        raise TypeError(f"Bandwidth type is not a float. Got {type(bandwidth)}.")

    if not isinstance(return_pdf, bool):
        raise TypeError(f"Return_pdf type is not a bool. Got {type(return_pdf)}.")

    device = image.device

    if image.dim() == 4:
        batch_size, n_channels, height, width = image.size()
    elif image.dim() == 3:
        batch_size = 1
        n_channels, height, width = image.size()
    elif image.dim() == 2:
        height, width = image.size()
        batch_size, n_channels = 1, 1
    else:
        raise ValueError(f"Input values must be a of the shape BxCxHxW, "
                         f"CxHxW or HxW. Got {image.shape}.")

    if bandwidth == -1.:
        bandwidth = (max - min) / n_bins
    if centers.numel() == 0:
        centers = min + bandwidth * (torch.arange(n_bins, device=device).float() + 0.5)
    centers = centers.reshape(-1, 1, 1, 1, 1)
    u = abs(image.unsqueeze(0) - centers) / bandwidth
    mask = (u <= 1).float()
    hist = torch.sum(((1 - u) * mask), dim=(-2, -1)).permute(1, 2, 0)
    if return_pdf:
        normalization = torch.sum(hist, dim=-1).unsqueeze(0) + 1e-10
        print(hist)
        pdf = hist / normalization
        return hist, pdf
    return hist, torch.zeros_like(hist, dtype=hist.dtype, device=device)
