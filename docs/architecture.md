# System Architecture

## Overview

This system computes novelty of concepts relative to structured reference frames. The core insight: novelty is not an intrinsic property but emerges from the relationship between new information and existing knowledge structures.

## Components

### Core Abstraction Layer (`core.py`)

Defines the fetch/parse loop pattern:

```
Focus → fetch(Focus, Frame) → Data → parse(Data, Focus, Frame) → (Termination | NextFocus)
```

**Termination conditions:**
- `INTEGRATED`: Concept already contained in or highly similar to frame contents
- `CONTRADICTS_ROOT`: Concept opposes foundational claims (high novelty)
- `ORTHOGONAL`: No connection to frame found after exhaustive search
- `DISRUPTS`: Would significantly shift stake allocations
- `MAX_ITER`: Iteration limit reached

### Reference Frame Structure

A reference frame consists of:

| Component | Description |
|-----------|-------------|
| Claim Hierarchies | Trees where depth indicates foundational importance |
| Tendency Stakes | Weighted allocations from drives (SURVIVAL, STATUS, MEANING, etc.) |
| Integrated Observations | Set of absorbed concepts |
| Similarity Function | Determines topical relatedness |
| Stance Detector | Determines PRO/CON/NEUTRAL positioning |

### Novelty Components

Four orthogonal dimensions measured by the loop:

1. **Integration Resistance (IR)**: Iterations required before termination. More iterations = harder to place in existing structures.

2. **Contradiction Depth (CD)**: When contradiction occurs, how deep in the hierarchy? Shallow contradictions (foundational claims) yield higher scores.

3. **Coverage Gap (CG)**: Fraction of hierarchies where the concept has no relevant position.

4. **Allocation Disruption (AD)**: Would integration shift tendency stake proportions?

**Composite Score:**
```
composite = (IR × CD × CG × AD)^(1/4)
```

Geometric mean ensures all dimensions contribute; a concept maximally novel on one dimension but zero on another scores lower than moderate novelty across all.

## Implementation Hierarchy

```
core.py              Abstract interfaces (NoveltyProbe, ReferenceFrame)
    ↓
wikidata_probe.py    Wikidata-backed implementation (WikidataProbe, WikidataFrame)
    ↓
wikidata.py          Raw Wikidata API queries and data structures
embeddings.py        Sentence embeddings and NLI for similarity/stance
```

### World Model (`world_model/`)

Structures for representing agent belief states:

- `tree.py`: Binary trees with PRO/CON children, stake-weighted nodes
- `agent.py`: Tendencies with allocations that sum to 1.0
- `attention.py`: Novelty-modulated attention allocation

## Data Flow

```
Input Concept
     ↓
[Resolve to Wikidata Q-ID]
     ↓
[Fetch: entity, metrics, ancestry, adjacent]
     ↓
[Parse: check containment, find claims, detect stance]
     ↓
[Decision: terminate or continue to adjacent node]
     ↓
[If terminate: compute novelty components]
     ↓
Output: NoveltyResult with termination reason and component scores
```

## Design Decisions

### Why Wikidata?
- Provides grounded knowledge graph topology (not just embedding distances)
- Sitelinks count serves as notability prior
- Explicit hierarchical relations (P31 instance-of, P279 subclass-of)
- Covers 100M+ items across domains

### Why the Loop Pattern?
The fetch/parse loop enables:
- Incremental exploration (not one-shot comparison)
- Cycle detection (track visited nodes)
- Attention-guided prioritization (salience ranking of adjacent nodes)
- Frame absorption during traversal

### Why Geometric Mean?
Ensures all novelty dimensions must be non-trivial for high composite score. Captures the intuition that truly novel concepts disrupt multiple aspects of a worldview simultaneously.
