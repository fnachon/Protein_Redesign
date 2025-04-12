"""
Adapted from Nakata, S., Mori, Y. & Tanaka, S. 
End-to-end protein–ligand complex structure generation with diffusion-based generative models.
BMC Bioinformatics 24, 233 (2023).
https://doi.org/10.1186/s12859-023-05354-5

Repository: https://github.com/shuyana/DiffusionProteinLigand
"""

import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ProteinReDiff.data import PDBDataModule
from ProteinReDiff.model import ProteinReDiffModel




def main(args):
    pl.seed_everything(args.seed, workers=True)
    if os.path.exists(args.save_dir):
        rmtree(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PDBDataModule.from_argparse_args(args)
    model = ProteinReDiffModel(args)

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        strategy=args.strategy,
        callbacks=[checkpoint_callback],
        default_root_dir=args.save_dir,
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = PDBDataModule.add_argparse_args(parser)
    parser = ProteinReDiffModel.add_argparse_args(parser)
    # Trainer args (explicitly added for Lightning 2.x compatibility)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_false")
    parser.add_argument("--max_epochs", type=int, default=-1)
    # Custom script args
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save_dir", type=Path, required=True)
    args = parser.parse_args()

    # https://github.com/Lightning-AI/lightning/issues/5558#issuecomment-1199306489
    warnings.filterwarnings("ignore", "Detected call of", UserWarning)

    main(args)
