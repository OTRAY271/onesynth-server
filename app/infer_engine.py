import os
from pathlib import Path

import numpy as np
import torch

from .spinvae.data import build
from .spinvae.evaluation.load import ModelLoader
from .spinvae.model import hierarchicalvae


class InferEngine:
    def __init__(
        self,
        model_path=os.path.join(
            os.path.dirname(__file__), "pretrained_weight/decoder_checkpoint.pth"
        ),
        remain_size=50,
        remain_rows=16,
    ) -> None:
        self.ae_model = self.__load_model(model_path)
        self.remain_idx = np.random.choice(257 * 251, remain_size, replace=False)
        self.remain_rows = remain_rows

    def __load_model(self, model_path: str) -> hierarchicalvae.HierarchicalVAE:
        model_config, train_config = ModelLoader.get_model_train_configs(Path(""))
        dataset = build.get_dataset(model_config, train_config)
        model_config.dim_z = -1
        ae_model = hierarchicalvae.HierarchicalVAE(
            model_config, train_config, dataset.preset_indexes_helper
        )
        ae_model.decoder.load_state_dict(torch.load(model_path))
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
