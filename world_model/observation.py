"""
Observation model - atomic units of information.
Sentence-sized, capped, no inherent polarity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid
import hashlib


MAX_OBSERVATION_BYTES = 280


@dataclass
class Observation:
    """
    An atomic unit of information.

    - Sentence-sized, capped at MAX_OBSERVATION_BYTES
    - No inherent polarity (pro/con is determined by position in tree)
    - Can appear in multiple trees with different positions
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.content.encode('utf-8')) > MAX_OBSERVATION_BYTES:
            encoded = self.content.encode('utf-8')[:MAX_OBSERVATION_BYTES]
            self.content = encoded.decode('utf-8', errors='ignore').rsplit(' ', 1)[0] + '...'

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()[:16]

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Observation({preview})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat(),
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        return cls(
            id=data["id"],
            content=data["content"],
            source_id=data.get("source_id", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ObservationStore:
    """Collection of observations with deduplication"""

    observations: dict[str, Observation] = field(default_factory=dict)
    _hash_index: dict[str, str] = field(default_factory=dict)

    def add(self, obs: Observation) -> tuple[Observation, bool]:
        """Add observation, deduplicating by content hash. Returns (observation, is_new)."""
        existing_id = self._hash_index.get(obs.content_hash)
        if existing_id:
            return self.observations[existing_id], False

        self.observations[obs.id] = obs
        self._hash_index[obs.content_hash] = obs.id
        return obs, True

    def get(self, obs_id: str) -> Optional[Observation]:
        return self.observations.get(obs_id)

    def all(self) -> list[Observation]:
        return list(self.observations.values())

    def __len__(self) -> int:
        return len(self.observations)
