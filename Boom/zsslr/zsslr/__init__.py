"""Modular zero-shot sign recognition research baseline."""
from .model import VisualEncoder, SemanticEncoder, ZSSLRCompatibilityModel
from .protocol import training_loss, predict, evaluate
from .attribution import FlipDifferenceOperator
