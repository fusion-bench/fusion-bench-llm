#!/usr/bin/env python3
"""
This is the CLI script that is executed when the user runs the `fusion_bench_llm` command.
The script is responsible for parsing the command-line arguments, loading the configuration file, and running the fusion algorithm.
"""
import importlib
import importlib.resources
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from fusion_bench.programs import BaseHydraProgram
from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)


def _get_default_config_path():
    for config_dir in ["fusion_bench_llm_config", "config"]:
        config_path = os.path.join(
            importlib.import_module("fusion_bench_llm").__path__[0], "..", config_dir
        )
        if os.path.exists(config_path) and os.path.isdir(config_path):
            return os.path.abspath(config_path)
    return None


@hydra.main(
    config_path=_get_default_config_path(),
    config_name="main",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    program: BaseHydraProgram = instantiate(cfg)
    program.run()


if __name__ == "__main__":
    main()
