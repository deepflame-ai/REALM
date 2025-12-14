import torch
import torch.nn as nn
from . import model_dict

def get_model(args):
    if args.model not in model_dict:
        raise ValueError(f"Unknown model: {args.model}")
    return model_dict[args.model](args)


class BCTNormalizer:
    def __init__(self, mean=None, std=None, safe_denominator=None, lam=0.1, eps=1e-8, num_chemical=8):
        self.lam = lam
        self.eps = eps
        self.mean = mean
        self.std = std
        self.num_chemical = num_chemical
        self.norm_eps = 1e-8

        self.safe_denominator = safe_denominator

    def BCT(self, x):
        return torch.log(torch.clamp(x, min=self.eps)) if self.lam == 0 else (torch.pow(torch.clamp(x, min=self.eps), self.lam) - 1) / self.lam

    def inverse_BCT(self, x):
        return torch.exp(x) if self.lam == 0 else torch.pow(self.lam * x + 1, 1 / self.lam)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        device = data.device
        data = data.clone()
        shape = [1, -1] + [1] * (data.dim() - 2)
        mean, safe_denominator = self.mean.reshape(shape), self.safe_denominator.reshape(shape)
        bct_part = data[:, :self.num_chemical]
        bct_part = self.BCT(bct_part)
        data[:, :self.num_chemical] = bct_part
        normed = (data - mean.to(device)) / safe_denominator
        return normed

    def decode(self, normed: torch.Tensor) -> torch.Tensor:
        shape = [1, -1] + [1] * (normed.dim() - 2)
        mean, safe_denominator = self.mean.reshape(shape), self.safe_denominator.reshape(shape)
        device = normed.device
        normed = normed * (safe_denominator) + mean.to(device)
        normed[:, :self.num_chemical] = self.inverse_BCT(normed[:, :self.num_chemical])
        return normed


class XDESolver(nn.Module):
    def __init__(self, args):
        super(XDESolver, self).__init__()
        self.model = get_model(args).to(args.device)
        self.normalizer = None
        self.num_chemical = args.num_chemical

    def forward(self, x, coords, time=None):
        if time is not None:
            return self.model(x, coords, time)
        else:
            return self.model(x, coords)

    def set_normalizer(self, mean, std, num_chemical=None):
        safe_denominator = torch.where(std < 1e-10, torch.ones_like(std), std + 1e-10)
        self.normalizer = BCTNormalizer(mean, std, safe_denominator, num_chemical=(num_chemical or self.num_chemical))

    def encoder(self, x):
        if self.normalizer is None:
            raise ValueError("Normalizer not set. Call set_normalizer() first.")
        return self.normalizer.encode(x)

    def decoder(self, x):
        if self.normalizer is None:
            raise ValueError("Normalizer not set. Call set_normalizer() first.")
        return self.normalizer.decode(x)