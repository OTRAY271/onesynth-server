from pathlib import Path

import numpy as np
import torch

from spinvae.evaluation.load import ModelLoader


class InferEngine:
    def __init__(
        self, model_path=Path("pretrained_weight"), remain_size=50, remain_rows=16
    ) -> None:
        loader = ModelLoader(model_path)
        loader.ae_model.decoder.eval()
        self.ae_model = loader.ae_model
        self.remain_idx = np.random.choice(257 * 251, remain_size, replace=False)
        self.remain_rows = remain_rows

    def __decode(self, z: torch.Tensor) -> torch.Tensor:
        zs = z.unsqueeze(0)
        zs = self.ae_model.unflatten_latent_values(zs)
        xs = self.ae_model.decoder(zs)[1]
        x = xs[0, 0]
        return torch.flatten(x)[self.remain_idx]

    def calc_vh_and_sigma(
        self, z: list[float]
    ) -> tuple[list[list[float]], list[float]]:
        z_tensor = torch.tensor(z)
        jacobian = torch.func.jacrev(self.__decode)(z_tensor).detach().numpy()
        _, sigma, vh = np.linalg.svd(jacobian, full_matrices=True)
        return vh[: self.remain_rows].tolist(), sigma[: self.remain_rows].tolist()
