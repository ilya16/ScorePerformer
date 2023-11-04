from .collators import (
    LMPerformanceCollator,
    MixedLMPerformanceCollator,
    LMScorePerformanceCollator,
    MixedLMScorePerformanceCollator,
)
from .datasets import (
    PerformanceDataset,
    LocalScorePerformanceDataset
)

DATASETS = {name: cls for name, cls in globals().items() if ".datasets." in str(cls)}
COLLATORS = {name: cls for name, cls in globals().items() if ".collators." in str(cls)}
