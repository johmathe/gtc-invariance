from configs.data.octa_modelnet import dataset_config
from configs.data_loader.bs64_val02 import data_loader_config
from configs.model.octa_cnn_bsp_10 import model_config
from configs.loss.cross_entropy import loss_config
from configs.optimizer.adam_5e5 import optimizer_config
from configs.scheduler.plateau import scheduler_config
from gtc.trainer import GTrainer


"""
MASTER CONFIG
"""

master_config = {
    "dataset": dataset_config,
    "data_loader": data_loader_config,
    "model": model_config,
    "loss": loss_config,
    "optimizer": optimizer_config,
    "scheduler": scheduler_config,
    "trainer": GTrainer,
    "seed": 4,
}


"""
LOGGER CONFIG
"""

from configs.logger.wandb_logger import logger_config
