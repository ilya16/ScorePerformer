from .base import Model
from .scoreperformer import (
    Performer,
    ScorePerformer,
    ScorePerformerEvaluator
)

MODELS = {name: cls for name, cls in globals().items() if ".model." in str(cls)}
EVALUATORS = {name: cls for name, cls in globals().items() if ".evaluator." in str(cls)}
