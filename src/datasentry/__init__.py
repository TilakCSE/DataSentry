"""DataSentry quality analysis and reproducible tabular experiments."""

from .experiment import ExperimentConfig, run_experiment
from .quality import profile_dataframe

__all__ = ["ExperimentConfig", "run_experiment", "profile_dataframe"]
__version__ = "0.1.0"
