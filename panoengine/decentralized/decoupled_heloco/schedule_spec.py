"""Pure-Python validation for the paper's H/fragment-offset schedule."""

def resolve_paper_schedule(num_fragments, sync_period=None, fragment_offsets=None):
    """Validate H and the unique offset assigned to each fragment."""
    period = num_fragments if sync_period is None else sync_period
    if type(period) is not int or period < num_fragments:
        raise ValueError("sync_period (H) must be an integer >= num_fragments")
    offsets = tuple(i * period // num_fragments for i in range(num_fragments)) if fragment_offsets is None else fragment_offsets
    if not isinstance(offsets, (list, tuple)) or len(offsets) != num_fragments:
        raise ValueError("fragment_offsets must contain one offset per fragment")
    if any(type(t) is not int or not 0 <= t < period for t in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("fragment_offsets must be distinct integers in [0, H)")
    return period, tuple(offsets)

