import argparse
import os
import shutil

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from detcon.datasets import VOCSSLDataModule
from detcon.models import DetConB


def main(cfg_path: str, cfg: DictConfig) -> None:
    pl.seed_everything(0, workers=True)
    module = DetConB(**cfg.module)
    datamodule = VOCSSLDataModule(**cfg.datamodule)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model=module, datamodule=datamodule)
    shutil.copyfile(cfg_path, os.path.join(trainer.logger.log_dir, "config.yaml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to config.yaml file"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    main(args.cfg, cfg)
