from pathlib import Path

import numpy as np
import torch

from spinvae.data import build
from spinvae.data.preset2d import Preset2d
from spinvae.evaluation.load import ModelLoader
from spinvae.model import hierarchicalvae


class InferEngine:
    SEARCH_LENGTH = 113 * 0.075

    def __init__(
        self,
        model_path=Path("pretrained_weight/decoder_checkpoint.pth"),
        remain_size=50,
        remain_rows=32,
    ) -> None:
        self.ae_model = self.__load_model(model_path)
        self.dataset = build.get_dataset(*ModelLoader.get_model_train_configs(None))
        self.remain_idx = np.random.choice(257 * 251, remain_size, replace=False)
        self.remain_rows = remain_rows

    def __load_model(self, model_path: Path) -> hierarchicalvae.HierarchicalVAE:
        model_config, train_config = ModelLoader.get_model_train_configs(Path(""))
        dataset = build.get_dataset(model_config, train_config)
        model_config.dim_z = -1
        ae_model = hierarchicalvae.HierarchicalVAE(
            model_config, train_config, dataset.preset_indexes_helper
        )
        ae_model.decoder.load_state_dict(torch.load(str(model_path)))
        ae_model.decoder.eval()

        return ae_model

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

    def calc_prob(self, sigma: list[float]) -> list[float]:
        sigma_np = np.array(sigma)
        sum_sigma = np.sum(sigma_np)
        if sum_sigma != 0:
            return sigma_np / sum_sigma
        else:
            return np.ones_like(sigma_np) / len(sigma_np)

    def infer_synth_params(self, z: torch.Tensor) -> list[float]:
        zs = z.unsqueeze(0)
        zs = self.ae_model.unflatten_latent_values(zs)
        u_out = self.ae_model.decoder.preset_decoder(zs)[0][0]
        preset2d = Preset2d(self.dataset, learnable_tensor_preset=u_out)
        params = list(preset2d.to_raw())
        return params

    def turn_around(self, vh: list[list[float]], prob: list[float]) -> list[float]:
        return vh[np.random.choice(len(prob), size=1, p=prob)[0]]

    def calc_z(
        self, position: float, basis: list[float], center_z: list[float]
    ) -> torch.Tensor:
        basis_tensor, center_z_tensor = torch.tensor(basis), torch.tensor(center_z)
        return center_z_tensor + position * InferEngine.SEARCH_LENGTH * basis_tensor
