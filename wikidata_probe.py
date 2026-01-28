"""
Wikidata Adapter for the Novelty Probe

Implements the core loop abstractions using Wikidata as the knowledge graph.

- WikidataProbe: fetch() queries Wikidata, parse() evaluates termination
- WikidataFrame: reference frame backed by Wikidata concepts
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Set

try:
    from .core import (
        NoveltyProbe,
        ReferenceFrame,
        Focus,
        ParseResult,
        Termination,
        Stance,
        Claim,
        NoveltyResult,
    )
    from .wikidata import (
        WikidataNoveltyInputs,
        WikidataEntity,
        GraphMetrics,
        AncestryPath,
        AdjacentNode,
        best_match,
        get_entity,
        get_graph_metrics,
        get_ancestry,
        get_labels,
        search_concept,
        get_adjacent_with_metadata,
        get_topically_related,
    )
except ImportError:
    from core import (
        NoveltyProbe,
        ReferenceFrame,
        Focus,
        ParseResult,
        Termination,
        Stance,
        Claim,
        NoveltyResult,
    )
    from wikidata import (
        WikidataNoveltyInputs,
        WikidataEntity,
        GraphMetrics,
        AncestryPath,
        AdjacentNode,
        best_match,
        get_entity,
        get_graph_metrics,
        get_ancestry,
        get_labels,
        search_concept,
        get_adjacent_with_metadata,
        get_topically_related,
    )


# =============================================================================
# Wikidata Reference Frame
# =============================================================================

@dataclass
class WikidataClaim(Claim):
    """A claim backed by a Wikidata concept."""
    qid: str = ""
    label: str = ""
    ancestry: List[str] = field(default_factory=list)


@dataclass
class WikidataFrame(ReferenceFrame):
    """
    A reference frame backed by Wikidata concepts.

    The frame contains:
    - A set of integrated Q-IDs (concepts already absorbed)
    - Claims derived from those concepts
    - Stake weights (derived from centrality/sitelinks)
    """
    integrated_qids: Set[str] = field(default_factory=set)
    claims: List[WikidataClaim] = field(default_factory=list)
    _total_stake: float = 0.0

    # Thresholds for termination decisions
    integration_threshold: float = 0.5  # Similarity above this = integrated
    contradiction_threshold: float = 0.6  # Stance confidence above this = contradiction

    def contains(self, content: Any) -> Tuple[bool, float]:
        """
        Check if content is already in the frame.

        Content can be:
        - A Q-ID string
        - A WikidataNoveltyInputs object
        - A text string (will be searched)
        """
        qid = self._to_qid(content)
        if qid is None:
            return False, 0.0

        # Direct containment
        if qid in self.integrated_qids:
            return True, 1.0

        # Check if content's direct relations overlap with frame
        try:
            entity = get_entity(qid)
            direct_relations = set(
                entity.subclass_of + entity.instance_of +
                entity.part_of + entity.uses
            )

            # Direct relation to frame concept
            direct_overlap = direct_relations & self.integrated_qids
            if direct_overlap:
                print(f"    [contains] Direct relation to frame: {direct_overlap}")
                return True, 0.9

            # Check if content is a direct subclass/instance of a frame concept
            # (This is meaningful - e.g., "digital currency" subclass of "currency")
            ancestry = get_ancestry(qid)
            for ancestor in ancestry.ancestors[:3]:  # Only first 3 levels
                if ancestor in self.integrated_qids:
                    print(f"    [contains] Direct ancestor in frame: {ancestor}")
                    return True, 0.85

            # Note: We deliberately DON'T check for shared ancestry with frame concepts
            # because that catches abstract categories like "entity" which don't
            # indicate meaningful semantic connection.

        except Exception as e:
            pass

        return False, 0.0

    def find_claims(self, content: Any) -> List[Tuple[Claim, float]]:
        """Find claims related to this content via shared ancestry."""
        qid = self._to_qid(content)
        if qid is None:
            return []

        try:
            ancestry = get_ancestry(qid)
        except:
            return []

        results = []
        for claim in self.claims:
            # Check for shared ancestors
            shared = set(ancestry.ancestors) & set(claim.ancestry)
            if shared or claim.qid in ancestry.ancestors:
                # Relevance based on how close the shared ancestor is
                if claim.qid in ancestry.ancestors:
                    idx = ancestry.ancestors.index(claim.qid)
                    relevance = 1.0 - (idx / max(len(ancestry.ancestors), 1))
                else:
                    relevance = len(shared) / max(len(ancestry.ancestors), 1)
                results.append((claim, relevance))

        return sorted(results, key=lambda x: -x[1])

    def detect_stance(self, content: Any, claim: Claim) -> Tuple[Stance, float]:
        """
        Detect stance based on graph structure.

        In Wikidata terms:
        - PRO: content is subclass/instance of claim, or shares positive relations
        - CON: content has "different from" or contradictory properties
        - NEUTRAL: related but no clear stance
        """
        qid = self._to_qid(content)
        if qid is None or not isinstance(claim, WikidataClaim):
            return Stance.NEUTRAL, 0.0

        try:
            entity = get_entity(qid)

            # Check subclass/instance relationships (PRO)
            if claim.qid in entity.subclass_of or claim.qid in entity.instance_of:
                return Stance.PRO, 0.9

            # Check if claim is in ancestry (PRO - supports hierarchy)
            ancestry = get_ancestry(qid)
            if claim.qid in ancestry.ancestors:
                return Stance.PRO, 0.8

            # Check for explicit "different from" (P1889) - would need to fetch
            # For now, use heuristic: very different centrality = potential conflict
            claim_metrics = get_graph_metrics(claim.qid)
            content_metrics = get_graph_metrics(qid)

            centrality_diff = abs(
                claim_metrics.centrality_ratio - content_metrics.centrality_ratio
            ) / max(claim_metrics.centrality_ratio, content_metrics.centrality_ratio, 1)

            if centrality_diff > 0.8:
                # Very different establishment levels - potential tension
                return Stance.CON, 0.5

            # Default: neutral but related
            return Stance.NEUTRAL, 0.6

        except:
            return Stance.NEUTRAL, 0.0

    def absorb(self, content: Any) -> "WikidataFrame":
        """Create new frame with content integrated."""
        qid = self._to_qid(content)
        if qid is None:
            return self

        new_integrated = self.integrated_qids | {qid}

        # Create claim for absorbed content
        try:
            entity = get_entity(qid)
            metrics = get_graph_metrics(qid)
            ancestry = get_ancestry(qid)

            new_claim = WikidataClaim(
                content=entity.label,
                depth=min(ancestry.depth, 10),
                stake=metrics.integration_score,  # More connected = more stake
                qid=qid,
                label=entity.label,
                ancestry=ancestry.ancestors,
            )
            new_claims = self.claims + [new_claim]
            new_stake = self._total_stake + new_claim.stake

        except:
            new_claims = self.claims
            new_stake = self._total_stake

        return WikidataFrame(
            integrated_qids=new_integrated,
            claims=new_claims,
            _total_stake=new_stake,
            integration_threshold=self.integration_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )

    def get_adjacent(self, content: Any, use_smart_expansion: bool = True) -> List[Any]:
        """
        Get adjacent concepts from Wikidata graph.

        Uses smart expansion that leverages Wikidata's built-in metadata
        (sitelinks, statements count) to filter noise and prioritize
        relevant neighbors.

        Args:
            content: The content to expand from
            use_smart_expansion: If True, use metadata-filtered expansion

        Returns:
            List of Q-IDs for adjacent concepts
        """
        qid = self._to_qid(content)
        if qid is None:
            return []

        try:
            if use_smart_expansion:
                # Use new smart expansion with Wikidata metadata filtering
                # This filters by sitelinks (notability) and prioritizes
                # meaningful relations (P31, P279) over noisy ones (P527)
                adjacent_nodes = get_topically_related(
                    qid,
                    reference_qids=list(self.integrated_qids)[:5],
                    max_results=10,
                )

                # Extract Q-IDs, excluding already integrated
                result = [
                    node.qid for node in adjacent_nodes
                    if node.qid not in self.integrated_qids
                ]

                if result:
                    print(f"    [smart expansion] {len(result)} relevant neighbors")
                    for node in adjacent_nodes[:3]:
                        print(f"      - {node.label} ({node.relation}, {node.sitelinks} sitelinks)")

                return result

            else:
                # Fallback: naive expansion (original behavior)
                entity = get_entity(qid)

                adjacent = []
                adjacent.extend(entity.subclass_of)
                adjacent.extend(entity.instance_of)
                adjacent.extend(entity.part_of)
                adjacent.extend(entity.has_parts[:5])
                adjacent.extend(entity.uses[:5])

                ancestry = get_ancestry(qid, max_depth=3)
                adjacent.extend(ancestry.ancestors[:3])

                seen = set()
                result = []
                for adj_qid in adjacent:
                    if adj_qid not in seen and adj_qid not in self.integrated_qids:
                        seen.add(adj_qid)
                        result.append(adj_qid)

                return result[:10]

        except Exception as e:
            print(f"    [get_adjacent] Error: {e}")
            return []

    @property
    def total_stake(self) -> float:
        return max(self._total_stake, 0.01)

    def _to_qid(self, content: Any) -> Optional[str]:
        """Convert various content types to Q-ID."""
        if isinstance(content, str):
            if content.startswith('Q') and content[1:].isdigit():
                return content
            # Search for it
            match = best_match(content)
            return match[0] if match else None
        elif isinstance(content, WikidataNoveltyInputs):
            return content.qid
        elif hasattr(content, 'qid'):
            return content.qid
        return None

    @classmethod
    def from_concepts(cls, concepts: List[str], **kwargs) -> "WikidataFrame":
        """Build a frame from a list of concept texts."""
        frame = cls(**kwargs)
        for concept in concepts:
            frame = frame.absorb(concept)
        return frame


# =============================================================================
# Wikidata Novelty Probe
# =============================================================================

@dataclass
class WikidataFetchResult:
    """Data fetched from Wikidata for a focus."""
    qid: str
    label: str
    entity: WikidataEntity
    metrics: GraphMetrics
    ancestry: AncestryPath
    adjacent: List[str]


class WikidataProbe(NoveltyProbe):
    """
    Novelty probe that fetches from Wikidata.

    The fetch/parse cycle:
    1. fetch() queries Wikidata for entity, metrics, ancestry
    2. parse() evaluates against frame and decides termination
    """

    def __init__(
        self,
        max_iterations: int = 15,
        integration_threshold: float = 0.7,
        contradiction_confidence: float = 0.6,
        disruption_threshold: float = 0.5,
    ):
        super().__init__(max_iterations)
        self.integration_threshold = integration_threshold
        self.contradiction_confidence = contradiction_confidence
        self.disruption_threshold = disruption_threshold
        self._visited: Set[str] = set()  # Track visited Q-IDs

    def measure(self, content: Any, frame: ReferenceFrame) -> NoveltyResult:
        """Run the novelty loop with cycle detection."""
        self._visited = set()  # Reset for new measurement
        return super().measure(content, frame)

    def fetch(self, focus: Focus, frame: ReferenceFrame) -> Optional[WikidataFetchResult]:
        """Fetch Wikidata data for the current focus."""
        content = focus.content

        # Resolve to Q-ID
        if isinstance(content, str):
            if content.startswith('Q') and content[1:].isdigit():
                qid = content
            else:
                match = best_match(content)
                if not match:
                    return None
                qid = match[0]
        elif hasattr(content, 'qid'):
            qid = content.qid
        else:
            return None

        # Track as visited
        self._visited.add(qid)

        try:
            entity = get_entity(qid)
            metrics = get_graph_metrics(qid)
            ancestry = get_ancestry(qid)

            # Get adjacent, excluding already visited
            if isinstance(frame, WikidataFrame):
                all_adjacent = frame.get_adjacent(qid)
                adjacent = [a for a in all_adjacent if a not in self._visited]
            else:
                adjacent = []

            return WikidataFetchResult(
                qid=qid,
                label=entity.label,
                entity=entity,
                metrics=metrics,
                ancestry=ancestry,
                adjacent=adjacent,
            )
        except Exception as e:
            print(f"  Fetch error for {qid}: {e}")
            return None

    def parse(
        self,
        data: Optional[WikidataFetchResult],
        focus: Focus,
        frame: ReferenceFrame,
    ) -> ParseResult:
        """
        Parse fetched data and decide termination.

        Termination conditions:
        1. INTEGRATED: content already in frame or highly similar
        2. CONTRADICTS_ROOT: opposes a foundational claim
        3. ORTHOGONAL: no connection to frame found
        4. DISRUPTS: would massively shift stake allocation
        """
        # No data = can't continue
        if data is None:
            return ParseResult.terminate(
                Termination.ORTHOGONAL,
                similarity_to_frame=0.0,
            )

        # Check integration
        is_contained, similarity = frame.contains(data.qid)
        if is_contained:
            return ParseResult.terminate(
                Termination.INTEGRATED,
                similarity_to_frame=similarity,
            )

        # High similarity without full containment = almost integrated
        if similarity >= self.integration_threshold:
            return ParseResult.terminate(
                Termination.INTEGRATED,
                similarity_to_frame=similarity,
            )

        # Find related claims and check for contradiction
        related_claims = frame.find_claims(data.qid)

        if related_claims:
            for claim, relevance in related_claims:
                stance, confidence = frame.detect_stance(data.qid, claim)

                if stance == Stance.CON and confidence >= self.contradiction_confidence:
                    # Found contradiction - check depth
                    return ParseResult.terminate(
                        Termination.CONTRADICTS_ROOT,
                        contradiction_depth=claim.depth,
                        stake_affected=claim.stake * relevance,
                        similarity_to_frame=similarity,
                    )

            # Related but not contradicting - check disruption potential
            max_stake = max(claim.stake * rel for claim, rel in related_claims)
            if max_stake >= self.disruption_threshold * frame.total_stake:
                return ParseResult.terminate(
                    Termination.DISRUPTS,
                    stake_affected=max_stake,
                    similarity_to_frame=similarity,
                )

        # No termination - continue to adjacent
        if data.adjacent:
            next_qid = data.adjacent[0]  # Take first adjacent
            next_focus = focus.expand_to(next_qid, via=data.label)

            return ParseResult.continue_to(
                focus=next_focus,
                absorbed=None,  # Don't absorb until we find connection
                similarity_to_frame=similarity,
            )

        # No adjacent = orthogonal
        return ParseResult.terminate(
            Termination.ORTHOGONAL,
            similarity_to_frame=similarity,
        )

    def _get_max_depth(self, frame: ReferenceFrame) -> int:
        """Get max claim depth from frame."""
        if isinstance(frame, WikidataFrame) and frame.claims:
            return max(c.depth for c in frame.claims)
        return 10


# =============================================================================
# Convenience Functions
# =============================================================================

def measure_novelty(
    concept: str,
    reference_concepts: List[str],
    max_iterations: int = 15,
    verbose: bool = True,
) -> NoveltyResult:
    """
    Measure novelty of a concept against a reference frame.

    Args:
        concept: The concept to measure (text or Q-ID)
        reference_concepts: List of concepts that form the reference frame
        max_iterations: Max probe iterations
        verbose: Print progress

    Returns:
        NoveltyResult with termination reason and component scores
    """
    if verbose:
        print(f"Building reference frame from {len(reference_concepts)} concepts...")

    frame = WikidataFrame.from_concepts(reference_concepts)

    if verbose:
        print(f"Frame has {len(frame.integrated_qids)} integrated concepts")
        print(f"Measuring novelty of: {concept}")
        print("-" * 50)

    probe = WikidataProbe(max_iterations=max_iterations)
    result = probe.measure(concept, frame)

    if verbose:
        print(f"\nTermination: {result.termination.value}")
        print(f"Iterations: {result.iterations}")
        print(f"Path: {' -> '.join(result.path) if result.path else '(direct)'}")
        print(f"\nComponents:")
        print(f"  integration_resistance: {result.integration_resistance:.3f}")
        print(f"  contradiction_depth: {result.contradiction_depth:.3f}")
        print(f"  coverage_gap: {result.coverage_gap:.3f}")
        print(f"  allocation_disruption: {result.allocation_disruption:.3f}")
        print(f"\nComposite: {result.composite:.3f}")

    return result


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test: Measure novelty of "blockchain" against a classical economics frame
    print("=" * 60)
    print("TEST: Novelty of 'blockchain' against classical concepts")
    print("=" * 60)

    classical_frame = [
        "money",
        "bank",
        "currency",
        "transaction",
        "ledger",
    ]

    result = measure_novelty("blockchain", classical_frame)

    print("\n" + "=" * 60)
    print("TEST: Novelty of 'water' against same frame")
    print("=" * 60)

    result2 = measure_novelty("water", classical_frame)

    print("\n" + "=" * 60)
    print("TEST: Novelty of 'digital currency' against same frame")
    print("=" * 60)

    result3 = measure_novelty("digital currency", classical_frame)
