"""
Attention module - novelty-modulated attention allocation.

This module implements a smooth curve function that describes how novelty
affects attention allocation across tendencies. Rather than a hard threshold,
novelty gradually captures attention as it increases.

The core insight: high novelty can "monopolize" attention regardless of
the agent's tendency profile, but the transition is smooth.

Formula:
    novelty_capture = sigmoid((novelty - midpoint) * steepness)
    effective_weight = (1 - capture) * base_allocation + capture * novelty_weight

Where:
- novelty_capture: How much attention novelty captures (0 to 1)
- base_allocation: The tendency's normal weight from AgentSet
- novelty_weight: Where attention flows under high novelty
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from .agent import Tendency, AgentSet


def sigmoid(x: float) -> float:
    """Standard sigmoid function, bounded (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class NoveltyAttentionCurve:
    """
    Defines how novelty modulates attention allocation.

    The curve is a sigmoid that maps novelty (0-1) to attention capture (0-1).

    Parameters:
        midpoint: Novelty value at which capture = 0.5 (default 0.5)
        steepness: How sharp the transition is (default 10)
        curiosity_bias: Extra weight CURIOSITY gets under high novelty (default 0.5)

    Behavior:
        - Low novelty (< midpoint): attention mostly follows tendency allocations
        - High novelty (> midpoint): attention shifts toward CURIOSITY
        - Extreme novelty (~1.0): CURIOSITY dominates, other tendencies dampened
    """

    midpoint: float = 0.5
    steepness: float = 10.0
    curiosity_bias: float = 0.5

    def capture(self, novelty: float) -> float:
        """
        Compute how much attention novelty captures.

        Args:
            novelty: Novelty score in range [0, 1]

        Returns:
            Attention capture coefficient in range [0, 1]
        """
        # Clamp novelty to valid range
        novelty = max(0.0, min(1.0, novelty))

        # Sigmoid centered at midpoint
        x = (novelty - self.midpoint) * self.steepness
        return sigmoid(x)

    def effective_allocations(
        self,
        base_allocations: dict[Tendency, float],
        novelty: float,
    ) -> dict[Tendency, float]:
        """
        Compute effective attention allocation given novelty.

        As novelty increases, allocation shifts toward CURIOSITY.
        Other tendencies are proportionally reduced.

        Args:
            base_allocations: Normal tendency weights (should sum to 1.0)
            novelty: Novelty score in range [0, 1]

        Returns:
            Modified allocations accounting for novelty (sums to 1.0)
        """
        capture = self.capture(novelty)

        if capture < 0.001:
            # No significant novelty effect
            return dict(base_allocations)

        result = {}

        # Under high novelty, CURIOSITY gets a boost
        # The boost comes from redistributing from other tendencies
        curiosity_boost = capture * self.curiosity_bias
        non_curiosity_dampening = curiosity_boost / (len(Tendency) - 1)

        for tendency, base_weight in base_allocations.items():
            if tendency == Tendency.CURIOSITY:
                # CURIOSITY gains under novelty
                effective = base_weight + curiosity_boost * (1 - base_weight)
            else:
                # Other tendencies are dampened proportionally
                effective = base_weight * (1 - curiosity_boost)

            result[tendency] = effective

        # Normalize to ensure sum = 1.0
        total = sum(result.values())
        if total > 0:
            result = {t: w / total for t, w in result.items()}

        return result

    def stake_weight(
        self,
        base_weight: float,
        tendency: Tendency,
        novelty: float,
    ) -> float:
        """
        Compute the effective stake weight for a tendency given novelty.

        This is a convenience method for computing a single tendency's
        effective weight without computing all allocations.

        Args:
            base_weight: The tendency's normal allocation
            tendency: Which tendency is staking
            novelty: Novelty score in range [0, 1]

        Returns:
            Effective stake weight
        """
        capture = self.capture(novelty)

        if capture < 0.001:
            return base_weight

        curiosity_boost = capture * self.curiosity_bias

        if tendency == Tendency.CURIOSITY:
            return base_weight + curiosity_boost * (1 - base_weight)
        else:
            return base_weight * (1 - curiosity_boost)


@dataclass
class AttentionState:
    """
    Tracks attention allocation for an agent during observation processing.

    This maintains both the base tendency allocations and the current
    novelty-modulated effective allocations.
    """

    agent_set: AgentSet
    curve: NoveltyAttentionCurve = field(default_factory=NoveltyAttentionCurve)
    current_novelty: float = 0.0
    _effective_cache: Optional[dict[Tendency, float]] = field(default=None, repr=False)

    @property
    def base_allocations(self) -> dict[Tendency, float]:
        """Get base tendency allocations from agent set."""
        return {t: a.allocation for t, a in self.agent_set.agents.items()}

    @property
    def effective_allocations(self) -> dict[Tendency, float]:
        """Get current novelty-modulated allocations."""
        if self._effective_cache is None:
            self._effective_cache = self.curve.effective_allocations(
                self.base_allocations,
                self.current_novelty,
            )
        return self._effective_cache

    def update_novelty(self, novelty: float):
        """Update current novelty level, invalidating cache."""
        self.current_novelty = max(0.0, min(1.0, novelty))
        self._effective_cache = None

    def get_stake_weight(self, tendency: Tendency) -> float:
        """Get the effective stake weight for a tendency."""
        return self.effective_allocations.get(tendency, 0.0)

    @property
    def novelty_capture(self) -> float:
        """How much attention novelty is currently capturing."""
        return self.curve.capture(self.current_novelty)

    @property
    def dominant_tendency(self) -> Tendency:
        """Which tendency currently has the highest effective allocation."""
        allocations = self.effective_allocations
        return max(allocations.keys(), key=lambda t: allocations[t])

    def describe(self) -> str:
        """Human-readable description of current attention state."""
        capture = self.novelty_capture
        dominant = self.dominant_tendency

        if capture < 0.2:
            novelty_desc = "low"
        elif capture < 0.5:
            novelty_desc = "moderate"
        elif capture < 0.8:
            novelty_desc = "high"
        else:
            novelty_desc = "extreme"

        return (
            f"Attention state: {novelty_desc} novelty (capture={capture:.1%}), "
            f"dominant={dominant.value} ({self.effective_allocations[dominant]:.1%})"
        )


# Preset curves for different agent profiles
EXPLORER_CURVE = NoveltyAttentionCurve(midpoint=0.3, steepness=8, curiosity_bias=0.7)
BALANCED_CURVE = NoveltyAttentionCurve(midpoint=0.5, steepness=10, curiosity_bias=0.5)
CONSERVATIVE_CURVE = NoveltyAttentionCurve(midpoint=0.7, steepness=12, curiosity_bias=0.3)
