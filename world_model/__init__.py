"""
World Model - Core structures for agent-relative novelty computation.

Adapted from the 'life' project for novelty research.
"""

from .observation import Observation, ObservationStore
from .agent import Tendency, Agent, AgentSet, DEFAULT_ALLOCATIONS
from .tree import Position, Stake, Node, Tree, TreeStore
from .attention import (
    NoveltyAttentionCurve,
    AttentionState,
    EXPLORER_CURVE,
    BALANCED_CURVE,
    CONSERVATIVE_CURVE,
)

__all__ = [
    "Observation",
    "ObservationStore",
    "Tendency",
    "Agent",
    "AgentSet",
    "DEFAULT_ALLOCATIONS",
    "Position",
    "Stake",
    "Node",
    "Tree",
    "TreeStore",
    "NoveltyAttentionCurve",
    "AttentionState",
    "EXPLORER_CURVE",
    "BALANCED_CURVE",
    "CONSERVATIVE_CURVE",
]
