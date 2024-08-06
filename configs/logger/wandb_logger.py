from torch_tools.config import Config
from torch_tools.logger import WBLogger


"""
LOGGING
"""
logger_config = Config(
    {
        "type": WBLogger,
        "params": {
            "project": "octahedral",
            "data_project": "gtc-invariance-scripts",
            "entity": "johmathe",
            "log_interval": 1,
            "watch_interval": 1,
            "plot_interval": 1,
            "end_plotter": None,
            "step_plotter": None,
        },
    }
)
