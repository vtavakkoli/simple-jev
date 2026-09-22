"""Dependency-free clients for explicit hosted JEV or local Simple Jev use."""
from .client import DecisionClient, DecisionError, validate_answers, validate_request

__version__ = "0.2.0"
__all__ = ["DecisionClient", "DecisionError", "validate_answers", "validate_request"]
