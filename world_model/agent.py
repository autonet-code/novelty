"""
Agent model - tendencies that stake on nodes in value trees.

Each agent represents a drive/tendency. Allocations determine relative
influence and sum to 1.0 across all agents.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Tendency(Enum):
    """Core tendencies - the agents that stake on nodes."""

    SURVIVAL = "survival"       # Physical safety, resource acquisition, risk mitigation
    STATUS = "status"           # Social standing, achievement, being valued
    MEANING = "meaning"         # Significance, impact, legacy, purpose
    CONNECTION = "connection"   # Relationships, community, being known
    AUTONOMY = "autonomy"       # Independence, self-determination, freedom
    COMFORT = "comfort"         # Ease, enjoyment, avoiding pain
    CURIOSITY = "curiosity"     # Knowledge, understanding, exploration


DEFAULT_ALLOCATIONS = {
    Tendency.SURVIVAL: 0.18,
    Tendency.STATUS: 0.12,
    Tendency.MEANING: 0.10,
    Tendency.CONNECTION: 0.20,
    Tendency.AUTONOMY: 0.12,
    Tendency.COMFORT: 0.18,
    Tendency.CURIOSITY: 0.10,
}


@dataclass
class Agent:
    """
    A tendency that stakes on nodes in value trees.

    Agents compete for influence through staking. Their allocations
    determine how much weight their stakes carry.
    """

    tendency: Tendency
    allocation: float = 0.0
    description: Optional[str] = None
    stakes_placed: int = 0
    stakes_validated: int = 0

    @property
    def id(self) -> str:
        return self.tendency.value

    @property
    def default_description(self) -> str:
        descriptions = {
            Tendency.SURVIVAL: "Physical safety, resource acquisition, risk mitigation",
            Tendency.STATUS: "Social standing, achievement, recognition, being valued",
            Tendency.MEANING: "Significance, impact, legacy, purpose beyond self",
            Tendency.CONNECTION: "Relationships, belonging, community, being known",
            Tendency.AUTONOMY: "Independence, self-determination, freedom from constraint",
            Tendency.COMFORT: "Ease, pleasure, avoiding pain, reducing friction",
            Tendency.CURIOSITY: "Knowledge, understanding, exploration, novelty",
        }
        return descriptions.get(self.tendency, "")

    @property
    def validation_rate(self) -> float:
        if self.stakes_placed == 0:
            return 0.0
        return self.stakes_validated / self.stakes_placed

    def __repr__(self):
        return f"Agent({self.tendency.value}, allocation={self.allocation:.2%})"

    def to_dict(self) -> dict:
        return {
            "tendency": self.tendency.value,
            "allocation": self.allocation,
            "description": self.description,
            "stakes_placed": self.stakes_placed,
            "stakes_validated": self.stakes_validated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        return cls(
            tendency=Tendency(data["tendency"]),
            allocation=data["allocation"],
            description=data.get("description"),
            stakes_placed=data.get("stakes_placed", 0),
            stakes_validated=data.get("stakes_validated", 0),
        )


@dataclass
class AgentSet:
    """
    The set of all agents for an entity.

    Manages allocations and provides methods for adjustment.
    Allocations always sum to 1.0.
    """

    agents: dict[Tendency, Agent] = field(default_factory=dict)
    calibrated: bool = False

    def __post_init__(self):
        if not self.agents:
            self._initialize_defaults()

    def _initialize_defaults(self):
        for tendency in Tendency:
            self.agents[tendency] = Agent(
                tendency=tendency,
                allocation=DEFAULT_ALLOCATIONS[tendency],
            )

    def get(self, tendency: Tendency) -> Agent:
        return self.agents[tendency]

    def all(self) -> list[Agent]:
        return list(self.agents.values())

    @property
    def total_allocation(self) -> float:
        return sum(a.allocation for a in self.agents.values())

    def normalize(self):
        total = self.total_allocation
        if total == 0:
            self._initialize_defaults()
            return
        for agent in self.agents.values():
            agent.allocation /= total

    def adjust_allocation(self, tendency: Tendency, delta: float):
        """Adjust an agent's allocation by delta, rebalancing others."""
        agent = self.agents[tendency]
        old_alloc = agent.allocation
        new_alloc = max(0.01, min(0.99, old_alloc + delta))

        actual_delta = new_alloc - old_alloc
        if abs(actual_delta) < 0.001:
            return

        agent.allocation = new_alloc

        others = [a for t, a in self.agents.items() if t != tendency]
        others_total = sum(a.allocation for a in others)

        if others_total > 0:
            for other in others:
                proportion = other.allocation / others_total
                other.allocation -= actual_delta * proportion

        self.normalize()

    def set_allocation(self, tendency: Tendency, value: float):
        current = self.agents[tendency].allocation
        self.adjust_allocation(tendency, value - current)

    def rebalance_by_performance(self, learning_rate: float = 0.1):
        """Shift allocations based on agent validation rates."""
        agents_with_stakes = [a for a in self.agents.values() if a.stakes_placed > 0]
        if len(agents_with_stakes) < 2:
            return

        avg_rate = sum(a.validation_rate for a in agents_with_stakes) / len(agents_with_stakes)

        for agent in agents_with_stakes:
            performance_delta = agent.validation_rate - avg_rate
            allocation_delta = performance_delta * learning_rate
            self.adjust_allocation(agent.tendency, allocation_delta)

        self.calibrated = True

    def __repr__(self):
        allocs = ", ".join(f"{t.value}={a.allocation:.0%}" for t, a in self.agents.items())
        return f"AgentSet({allocs})"

    def to_dict(self) -> dict:
        return {
            "agents": {t.value: a.to_dict() for t, a in self.agents.items()},
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSet":
        agent_set = cls(agents={})
        for tendency_str, agent_data in data["agents"].items():
            tendency = Tendency(tendency_str)
            agent_set.agents[tendency] = Agent.from_dict(agent_data)
        agent_set.calibrated = data.get("calibrated", False)
        return agent_set

    @classmethod
    def with_profile(cls, profile: dict[Tendency, float]) -> "AgentSet":
        """Create agent set with custom allocations."""
        agent_set = cls()
        for tendency, allocation in profile.items():
            agent_set.agents[tendency].allocation = allocation
        agent_set.normalize()
        agent_set.calibrated = True
        return agent_set
