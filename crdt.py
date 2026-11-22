"""
crdt.py -- Convergence core of the Mesh + CRDT sensor network.

This is the file everything else in the backend exists to serve:
    mesh.py        delivers deltas/snapshots to NodeState.merge()
    simulation.py  orchestrates scenarios that exercise NodeState.merge()
    crdt.py        defines what "converged" even means (NodeState.state_hash())

THE THREE PROPERTIES (read this before reading any code)
--------------------------------------------------------
Every merge() in this module must be:

    Commutative   merge(A, B) == merge(B, A)
                  -> messages may arrive in ANY order (multipath flooding
                     reorders everything).
    Associative   merge(merge(A, B), C) == merge(A, merge(B, C))
                  -> updates may be grouped/batched/relayed in any way
                     (node X may hand us B and C already merged together).
    Idempotent    merge(A, A) == A
                  -> duplicates are harmless (flooding delivers the same
                     update over several paths; we never need exactly-once).

If ANY of the three is violated, two nodes that saw the same set of updates
can end up in different states, and the convergence demo is meaningless.
Mathematically: state is a join-semilattice and merge is its least upper
bound. In practice: merge = "union of knowledge, with a deterministic rule
for conflicts".

TRAINING PATH (matches the six-step study plan)
-----------------------------------------------
  1. This docstring                       (why the properties matter)
  2. LamportClock                         (ordering without wall-clock time)
  3. GSet, PNCounter, LWWRegister/LWWMap, MembershipSet   (each merge())
  4. NodeState                            (composition + state_hash())
  5. GSet.compact()                       (the one tricky, non-CRDT method)
  6. The tests at the bottom              (run: python crdt.py  or  pytest crdt.py)

Design rules used throughout
  * merge() mutates self in place and returns True if self changed
    (useful for anti-entropy: "did that link teach me anything?").
  * merge() never mutates `other`.
  * Nothing here touches the network, the clock, or the disk. It is pure
    logic, so it is deterministic and unit-testable.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

ReadingKey = Tuple[str, int]   # (node_id, seq): globally unique, immutable identity
Reading = Tuple[int, float]    # (lamport_ts, value)


# ---------------------------------------------------------------------------
# 2. LamportClock -- logical time, because wall clocks drift
# ---------------------------------------------------------------------------
class LamportClock:
    """Logical clock (Lamport, 1978).

    Sensors have no NTP, so their wall clocks disagree. If "last write wins"
    were decided by wall-clock time, two nodes could disagree about which
    write was later and never converge. A Lamport clock encodes *causality*
    instead: if event X could have influenced event Y, then ts(X) < ts(Y).

    Rules:
        local event / before sending   -> tick():        time += 1
        on receiving remote timestamp  -> observe(t):    time = max(time, t) + 1

    Note: it is NOT real-time accurate. Concurrent events get arbitrary (but
    consistent) order; that is fine, we only need a deterministic one.
    """

    def __init__(self, start: int = 0) -> None:
        if start < 0:
            raise ValueError("clock cannot start negative")
        self.time = start

    def tick(self) -> int:
        self.time += 1
        return self.time

    def observe(self, remote_time: int) -> int:
        self.time = max(self.time, remote_time) + 1
        return self.time


# ---------------------------------------------------------------------------
# 3a. GSet -- grow-only set of sensor readings
# ---------------------------------------------------------------------------
class GSet:
    """Grow-only set of readings, keyed by (node_id, seq).

    The key IS the identity of a reading, so a reading is written exactly once
    by exactly one node. merge() is dict union -- the simplest possible CRDT
    and the easiest to prove correct by hand:
        union is commutative, associative and idempotent.

    Defensive rule: if two different values ever appear under the same key
    (a bug or a forged message), we keep the greater (ts, value) tuple. That
    is arbitrary but deterministic and order-independent, so even a corrupt
    input cannot break commutativity.
    """

    def __init__(self) -> None:
        self.items: Dict[ReadingKey, Reading] = {}

    def add(self, node_id: str, seq: int, ts: int, value: float) -> bool:
        key = (node_id, seq)
        incoming: Reading = (ts, value)
        current = self.items.get(key)
        if current is None or incoming > current:
            self.items[key] = incoming
            return True
        return False

    def merge(self, other: "GSet") -> bool:
        changed = False
        for (node_id, seq), (ts, value) in other.items.items():
            changed |= self.add(node_id, seq, ts, value)
        return changed

    def __len__(self) -> int:
        return len(self.items)

    def compact(self, keep_per_node: int) -> int:
        """Checkpoint-triggered retention: keep only the newest `keep_per_node`
        readings (by seq) of each node. Returns how many were evicted.

        THIS IS NOT A CRDT OPERATION. Deleting is not monotone, so it is not
        covered by the merge guarantees. It is a policy you apply only at a
        *checkpoint*:

          * Call it only when the network is verified quiescent, i.e. every
            node reports the same state_hash(). Then every node applies the
            same deterministic rule to the same data and they all land on the
            same compacted state.
          * If a node was offline during the checkpoint and later reconnects
            still holding the old readings, merge() simply re-introduces them
            (merge is still idempotent: it only ever adds). Everyone that
            receives them converges to the same *larger* set, no divergence
            and no corruption. The next checkpoint compacts them again,
            because compact() is a pure function of the state it is given.
          * What you must NOT do is compact on some nodes continuously while
            merges are still flowing: a node could evict a reading its peers
            still hold and the network would ping-pong it back and forth.
        """
        if keep_per_node < 0:
            raise ValueError("keep_per_node must be >= 0")
        seqs_by_node: Dict[str, List[int]] = {}
        for node_id, seq in self.items:
            seqs_by_node.setdefault(node_id, []).append(seq)
        evicted = 0
        for node_id, seqs in seqs_by_node.items():
            seqs.sort()
            doomed = seqs if keep_per_node == 0 else seqs[:-keep_per_node]
            for seq in doomed:
                del self.items[(node_id, seq)]
                evicted += 1
        return evicted


# ---------------------------------------------------------------------------
# 3b. PNCounter -- per-node increment/decrement slots
# ---------------------------------------------------------------------------
class PNCounter:
    """Positive-Negative counter, e.g. "total flood alerts currently active".

    State: two maps, p[node] (increments) and n[node] (decrements).
    value = sum(p) - sum(n).

    Each node writes ONLY its own slot, and a slot only ever grows, so for a
    given slot "larger" always means "more recent". merge() therefore takes the
    element-wise MAX.

    Why max and not sum? Because updates are delivered at-least-once. Summing
    would count a duplicated update twice (breaks idempotence); max counts it
    once no matter how many copies arrive, in any order.
    """

    def __init__(self) -> None:
        self.p: Dict[str, int] = {}
        self.n: Dict[str, int] = {}

    def increment(self, node_id: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("use decrement() for negative changes")
        self.p[node_id] = self.p.get(node_id, 0) + amount

    def decrement(self, node_id: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self.n[node_id] = self.n.get(node_id, 0) + amount

    @property
    def value(self) -> int:
        return sum(self.p.values()) - sum(self.n.values())

    @staticmethod
    def _max_merge(mine: Dict[str, int], theirs: Dict[str, int]) -> bool:
        changed = False
        for node_id, count in theirs.items():
            if count > mine.get(node_id, 0):
                mine[node_id] = count
                changed = True
        return changed

    def merge(self, other: "PNCounter") -> bool:
        changed_p = self._max_merge(self.p, other.p)
        changed_n = self._max_merge(self.n, other.n)
        return changed_p or changed_n


# ---------------------------------------------------------------------------
# 3c. LWWRegister / LWWMap -- last-writer-wins values (e.g. battery level)
# ---------------------------------------------------------------------------
class LWWRegister:
    """Single value; the write with the highest (ts, writer) wins.

    ts is a Lamport timestamp. writer (a node id) is the tiebreaker: two nodes
    can write with the SAME Lamport time concurrently, and every replica must
    still pick the SAME winner. Comparing writer ids gives a total order.

    A third-level tiebreak on the serialized value keeps merge commutative even
    if a buggy/forged writer sends two different values with identical
    (ts, writer).
    """

    __slots__ = ("value", "ts", "writer")

    def __init__(self, value: Any, ts: int, writer: str) -> None:
        self.value = value
        self.ts = ts
        self.writer = writer

    def rank(self) -> Tuple[int, str, str]:
        return (self.ts, self.writer, json.dumps(self.value, sort_keys=True))

    def merge(self, other: "LWWRegister") -> bool:
        if other.rank() > self.rank():
            self.value, self.ts, self.writer = other.value, other.ts, other.writer
            return True
        return False


class LWWMap:
    """Map of key -> LWWRegister. Each key merges independently."""

    def __init__(self) -> None:
        self.registers: Dict[str, LWWRegister] = {}

    def set(self, key: str, value: Any, ts: int, writer: str) -> bool:
        incoming = LWWRegister(value, ts, writer)
        current = self.registers.get(key)
        if current is None:
            self.registers[key] = incoming
            return True
        return current.merge(incoming)

    def get(self, key: str, default: Any = None) -> Any:
        reg = self.registers.get(key)
        return reg.value if reg is not None else default

    def merge(self, other: "LWWMap") -> bool:
        changed = False
        for key, reg in other.registers.items():
            changed |= self.set(key, reg.value, reg.ts, reg.writer)
        return changed


# ---------------------------------------------------------------------------
# 3d. MembershipSet -- "which nodes have I ever heard of"
# ---------------------------------------------------------------------------
class MembershipSet:
    """Grow-only set of node ids: merge is plain set union.

    Grow-only BY DESIGN: a node is never marked "gone". In a partitioned mesh,
    "I can't reach X" is indistinguishable from "X is dead", so we only record
    presence-ever. (Removal needs an OR-Set with tombstones -- more machinery
    than this project needs.)
    """

    def __init__(self) -> None:
        self.members: Set[str] = set()

    def add(self, node_id: str) -> bool:
        if node_id in self.members:
            return False
        self.members.add(node_id)
        return True

    def merge(self, other: "MembershipSet") -> bool:
        before = len(self.members)
        self.members |= other.members
        return len(self.members) != before

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.members


# ---------------------------------------------------------------------------
# 4. NodeState -- the four CRDTs composed into one world-state
# ---------------------------------------------------------------------------
class NodeState:
    """Everything one node knows about the world.

    merge() invents no logic of its own: it delegates to each sub-CRDT's merge.
    A product of semilattices is a semilattice, so if each part is correct the
    whole is correct. That is the entire correctness argument.

    Shared (replicated, hashed):     readings, alert_count, battery, membership
    Per-node bookkeeping (NOT hashed): evicted_readings, rejected_messages, clock
        These describe *this node's* history (what it compacted, how many bad
        messages it dropped, its local logical time). They legitimately differ
        between nodes that hold identical world-state, so including them would
        make two fully converged nodes report different hashes.
    """

    def __init__(self, node_id: str) -> None:
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")
        self.node_id = node_id
        self.clock = LamportClock()
        self.readings = GSet()
        self.alert_count = PNCounter()
        self.battery = LWWMap()
        self.membership = MembershipSet()
        self._next_seq = 0                 # local only; never reused, survives compaction
        self.evicted_readings = 0          # bookkeeping
        self.rejected_messages = 0         # bookkeeping
        self.membership.add(node_id)

    # -- local operations: always succeed, need no network -------------------
    def record_reading(self, value: float) -> ReadingKey:
        value = float(value)
        if not math.isfinite(value):       # NaN would break ordering and hashing
            raise ValueError("reading must be a finite number")
        ts = self.clock.tick()
        seq = self._next_seq
        self._next_seq += 1
        self.readings.add(self.node_id, seq, ts, value)
        return (self.node_id, seq)

    def raise_alert(self) -> None:
        self.clock.tick()
        self.alert_count.increment(self.node_id)

    def clear_alert(self) -> None:
        self.clock.tick()
        self.alert_count.decrement(self.node_id)

    def set_battery(self, level: float) -> None:
        level = float(level)
        if not math.isfinite(level):
            raise ValueError("battery level must be finite")
        ts = self.clock.tick()
        self.battery.set(self.node_id, level, ts, self.node_id)

    def note_rejected(self) -> None:
        """Called by the mesh layer when a message fails HMAC/validation."""
        self.rejected_messages += 1

    def compact(self, keep_per_node: int) -> int:
        """Checkpoint-only retention (see GSet.compact for the safety rules)."""
        evicted = self.readings.compact(keep_per_node)
        self.evicted_readings += evicted
        return evicted

    # -- replication ---------------------------------------------------------
    def merge(self, other: "NodeState") -> bool:
        """Fold another node's state into ours. Returns True if we learned anything."""
        results = [                         # list, not `or`: never short-circuit a merge
            self.readings.merge(other.readings),
            self.alert_count.merge(other.alert_count),
            self.battery.merge(other.battery),
            self.membership.merge(other.membership),
        ]
        self.clock.observe(other.clock.time)
        return any(results)

    # -- convergence identity ------------------------------------------------
    def shared_state(self) -> Dict[str, Any]:
        """Canonical, order-independent view of the replicated data only."""
        return {
            "readings": [[n, s, ts, v] for (n, s), (ts, v) in sorted(self.readings.items.items())],
            "alert_p": dict(sorted(self.alert_count.p.items())),
            "alert_n": dict(sorted(self.alert_count.n.items())),
            "battery": {k: [r.value, r.ts, r.writer] for k, r in sorted(self.battery.registers.items())},
            "membership": sorted(self.membership.members),
        }

    def state_hash(self) -> str:
        """Single source of truth for 'have these nodes converged?'."""
        blob = json.dumps(self.shared_state(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # -- (de)serialization for the mesh layer --------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = self.shared_state()
        d["node_id"] = self.node_id
        d["clock"] = self.clock.time
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeState":
        """Rebuild a state from a wire payload. Structural checks only;
        authentication (HMAC) and schema validation live in the mesh layer."""
        try:
            state = cls(d["node_id"])
            state.clock = LamportClock(int(d["clock"]))
            for n, s, ts, v in d["readings"]:
                state.readings.add(str(n), int(s), int(ts), float(v))
            state.alert_count.p = {str(k): int(v) for k, v in d["alert_p"].items()}
            state.alert_count.n = {str(k): int(v) for k, v in d["alert_n"].items()}
            for k, (v, ts, w) in d["battery"].items():
                state.battery.set(str(k), v, int(ts), str(w))
            for m in d["membership"]:
                state.membership.add(str(m))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed state payload: {exc}") from exc
        return state


# ===========================================================================
# 6. Tests -- run with `python crdt.py` (no dependencies) or `pytest crdt.py`
# ===========================================================================
def _copy(state: NodeState) -> NodeState:
    return NodeState.from_dict(json.loads(json.dumps(state.to_dict())))


def _three_writers() -> List[NodeState]:
    a, b, c = NodeState("A"), NodeState("B"), NodeState("C")
    for v in (1.0, 2.0, 3.0):
        a.record_reading(v)
    b.record_reading(10.0)
    b.record_reading(11.0)
    b.raise_alert()
    b.set_battery(80)
    c.record_reading(20.0)
    c.raise_alert()
    c.raise_alert()
    c.clear_alert()
    c.set_battery(55)
    a.set_battery(90)
    return [a, b, c]


def test_lamport_trace() -> None:
    # Two nodes write independently, then exchange messages.
    a, b = LamportClock(), LamportClock()
    assert a.tick() == 1                    # A: local event
    assert b.tick() == 1                    # B: concurrent local event
    assert b.tick() == 2                    # B: second local event
    sent_by_a = a.tick()                    # A sends at time 2
    assert sent_by_a == 2
    assert b.observe(sent_by_a) == 3        # B: max(2, 2) + 1
    sent_by_b = b.tick()                    # B replies at time 4
    assert a.observe(sent_by_b) == 5        # A: max(2, 4) + 1
    assert sent_by_a < 3 < sent_by_b < 5    # cause always precedes effect


def test_gset_merge_properties() -> None:
    x, y, z = GSet(), GSet(), GSet()
    x.add("A", 0, 1, 1.0)
    y.add("B", 0, 1, 2.0)
    z.add("A", 0, 1, 1.0)
    z.add("C", 0, 5, 3.0)

    def m(*sets: GSet) -> Dict[ReadingKey, Reading]:
        out = GSet()
        for s in sets:
            out.merge(s)
        return out.items

    assert m(x, y) == m(y, x)                              # commutative
    assert m(m_set(x, y), z) == m(x, m_set(y, z))          # associative
    assert m(x, x) == x.items                              # idempotent


def m_set(*sets: GSet) -> GSet:
    out = GSet()
    for s in sets:
        out.merge(s)
    return out


def test_pncounter_max_not_sum() -> None:
    a, b = PNCounter(), PNCounter()
    a.increment("A", 2)
    b.merge(a)
    b.merge(a)                              # duplicate delivery
    b.merge(a)
    assert b.value == 2                     # sum-based merge would give 6
    b.decrement("B")
    a.merge(b)
    assert a.value == b.value == 1


def test_lww_tiebreak_is_deterministic() -> None:
    m1, m2 = LWWMap(), LWWMap()
    m1.set("battery", 50, ts=7, writer="A")
    m2.set("battery", 60, ts=7, writer="B")  # same Lamport time, different writer
    left, right = LWWMap(), LWWMap()
    left.merge(m1)
    left.merge(m2)
    right.merge(m2)
    right.merge(m1)
    assert left.get("battery") == right.get("battery") == 60   # "B" > "A"


def test_membership_is_grow_only() -> None:
    a, b = MembershipSet(), MembershipSet()
    a.add("X")
    b.add("Y")
    a.merge(b)
    assert "X" in a and "Y" in a
    assert not a.merge(b)                   # second merge changes nothing


def test_nodestate_converges_in_every_merge_order() -> None:
    parts = _three_writers()
    hashes = set()
    for order in itertools.permutations(parts):
        observer = NodeState("Z")
        for p in order:
            observer.merge(p)
            observer.merge(p)               # duplicates on purpose
        hashes.add(observer.state_hash())
    assert len(hashes) == 1


def test_hash_ignores_bookkeeping() -> None:
    a, b = _three_writers()[0], _three_writers()[0]
    a.evicted_readings, a.rejected_messages = 99, 7
    a.clock.time += 1000
    assert a.state_hash() == b.state_hash()


def test_compact_then_late_reintroduction() -> None:
    a, b, c = _three_writers()
    for x in (a, b, c):                     # full anti-entropy: quiescent network
        for y in (a, b, c):
            x.merge(y)
    assert len({n.state_hash() for n in (a, b, c)}) == 1
    full = a.state_hash()

    a.compact(keep_per_node=1)              # checkpoint on A and B ...
    b.compact(keep_per_node=1)              # ... but C was offline and skipped it
    assert a.state_hash() == b.state_hash() != full
    assert a.evicted_readings > 0

    a.merge(c)                              # C reconnects with the old readings
    b.merge(c)
    assert a.state_hash() == b.state_hash() == c.state_hash() == full   # still converges

    for n in (a, b, c):                     # next checkpoint: same deterministic result
        n.compact(keep_per_node=1)
    assert len({n.state_hash() for n in (a, b, c)}) == 1


def test_wire_roundtrip() -> None:
    a = _three_writers()[1]
    assert _copy(a).state_hash() == a.state_hash()
    try:
        NodeState.from_dict({"node_id": "Q"})
    except ValueError:
        pass
    else:
        raise AssertionError("malformed payload should be rejected")


def test_non_finite_values_rejected() -> None:
    n = NodeState("A")
    for bad in (float("nan"), float("inf")):
        try:
            n.record_reading(bad)
        except ValueError:
            continue
        raise AssertionError("non-finite reading accepted")


# EXERCISE (study step 6) -- write this one yourself, then delete this comment:
#
#   def test_gset_three_way_all_orders():
#       """Build three GSets with overlapping keys, merge them pairwise in
#       every possible grouping and order (itertools.permutations, and both
#       ((x+y)+z) and (x+(y+z)) groupings) and assert identical final items.
#       This is your proof of associativity."""


def run_self_checks() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    return len(tests)


def demo_partition_and_gossip(seed: Optional[int] = 7) -> None:
    """Toy version of the project demo: independent writes, then random
    pairwise exchanges until every replica reports the same hash."""
    rng = random.Random(seed)
    nodes = [NodeState(f"S{i}") for i in range(5)]
    for i, n in enumerate(nodes):        # partitioned: each writes alone
        for _ in range(rng.randint(1, 4)):
            n.record_reading(rng.uniform(0, 10))
        n.set_battery(rng.randint(20, 100))
        if i % 2:
            n.raise_alert()
    rounds = 0
    while len({n.state_hash() for n in nodes}) > 1:
        rounds += 1
        x, y = rng.sample(nodes, 2)      # random link comes up, random order
        x.merge(y)
        y.merge(x)
    print(f"  5 partitioned nodes converged after {rounds} random exchanges")
    print(f"  final hash {nodes[0].state_hash()[:16]}...  "
          f"readings={len(nodes[0].readings)}  active_alerts={nodes[0].alert_count.value}")


if __name__ == "__main__":
    print("Running CRDT self-checks:")
    count = run_self_checks()
    print(f"{count} checks passed.\n")
    print("Partition + gossip demo:")
    demo_partition_and_gossip()
