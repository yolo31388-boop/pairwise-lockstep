"""Deterministic helpers for pairwise lockstep.

Everything that feeds the simulation must produce bit-identical results on
every device, so this module follows these rules:

* simulation maths is integer fixed-point (1/1000), never platform ``float``;
* inputs are ordered by ``(frame, player)`` before the simulation sees them;
* the RNG seed is derived from the room id negotiated by the peers, never
  from a clock;
* fixed-point multiplication saturates to the signed 32-bit range instead of
  wrapping (wrapping turned damage values negative);
* inputs older than the current logical frame are rejected, and missing
  player inputs for a frame are filled with empty inputs, so a retransmitted
  packet can never be replayed into a future frame and teleport an actor;
* logical frames advance in whole steps and are not driven by render deltas.
"""

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json

SCALE = 1000
INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1


def _saturate(value):
    """Clamp an integer to the signed 32-bit range."""
    if value > INT32_MAX:
        return INT32_MAX
    if value < INT32_MIN:
        return INT32_MIN
    return value


def fixed(value):
    """Convert a number to fixed-point (thousandths).

    ``float`` input is converted through its decimal string so the result
    does not depend on the platform's binary floating-point rounding.
    """
    if isinstance(value, Decimal):
        decimal_value = value
    else:
        decimal_value = Decimal(str(value))
    scaled = (decimal_value * SCALE).to_integral_value(rounding=ROUND_HALF_UP)
    return _saturate(int(scaled))


def fixed_mul(a, b):
    """Multiply two fixed-point values, saturating instead of wrapping.

    The intermediate product is held to the signed 32-bit range before it is
    rescaled. Wrapping that product around (which is what the old code did)
    is how damage values came out negative; here it saturates instead.
    """
    product = int(a) * int(b)
    if product > INT32_MAX:
        return INT32_MAX
    if product < INT32_MIN:
        return INT32_MIN
    # Round half away from zero, matching ``fixed``.
    if product >= 0:
        return (product + SCALE // 2) // SCALE
    return -((-product + SCALE // 2) // SCALE)


def sort_inputs(inputs):
    """Return inputs in canonical order: frame first, then player id."""
    return sorted(
        inputs,
        key=lambda item: (item.get("frame", 0), item.get("player", 0)),
    )


def detect_desync(hashes):
    """Return the indices of peers whose state hash differs from the majority.

    Ties are resolved in favour of the earliest hash in ``hashes`` so the
    result is deterministic on every peer.
    """
    if not hashes:
        return []
    counts = Counter(hashes)
    best_count = max(counts.values())
    majority = next(value for value in hashes if counts[value] == best_count)
    return [index for index, value in enumerate(hashes) if value != majority]


def room_seed(room_id):
    """Derive the shared RNG seed from the room id negotiated by the peers.

    The same room always yields the same seed on every device and in every
    replay; the digest is stable across processes and architectures.
    """
    material = b"pairwise-lockstep/v1:" + str(room_id).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:4], "big")


def advance_frame(frame, steps=1):
    """Advance the logical frame by whole simulation steps.

    The render loop decides *when* to call this, but the logical frame only
    ever moves by complete fixed steps, so a fluctuating frame rate cannot
    drag the simulation along with it.
    """
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return int(frame) + steps


def state_hash(state, inputs):
    """Hash simulation state plus its inputs into a canonical hex digest.

    Dict keys are sorted and dict-shaped inputs are canonicalised with
    :func:`sort_inputs`, so differing network arrival order cannot by itself
    produce a false desync.
    """

    def _encode(value):
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): _encode(item)
                for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(value, (list, tuple)):
            return [_encode(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(_encode(item) for item in value)
        return value

    canonical_inputs = inputs
    if inputs and all(
        isinstance(item, dict) and "frame" in item and "player" in item
        for item in inputs
    ):
        canonical_inputs = sort_inputs(inputs)

    payload = json.dumps(
        [_encode(state), _encode(canonical_inputs)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def accept_input(input_frame, current_frame, window=0):
    """Return whether an input may still be applied to its own frame.

    Inputs for frames that have already been simulated are stale: due to
    retransmission they may arrive late, and inserting them into a future
    frame would make an actor jump. They are dropped and an empty input is
    used for that player instead.
    """
    window = int(window)
    if window < 0:
        raise ValueError("window must be non-negative")
    return int(input_frame) >= int(current_frame) - window


def fill_missing_inputs(inputs, frame, players, empty=None):
    """Return one input per player for ``frame``, padding gaps with empties.

    ``inputs`` may contain inputs for other frames; they are ignored. Empty
    inputs are built from the optional ``empty`` mapping plus frame/player.
    """
    empty = dict(empty or {})
    by_player = {}
    for item in sort_inputs(inputs):
        if item.get("frame") != frame:
            continue
        by_player.setdefault(item.get("player"), item)

    padded = []
    for player in sorted(players):
        item = by_player.get(player)
        if item is None:
            item = {"frame": frame, "player": player}
            item.update(empty)
        padded.append(dict(item))
    return padded
