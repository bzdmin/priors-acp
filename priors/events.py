"""ACP event decoding and event sources.

Signatures and field layouts here were derived empirically from the decoded
log stream and cross-checked against per-event counts, because the ACP
contract sits behind an ERC-1967 proxy and fetching the implementation ABI
requires network access we do not want on the replay path.

Verification performed (2026-09-01, 224,064 logs, blocks 44,427,013-50,749,122):

  * Counts match the ABI-decoded totals in ``acp_decoded.json`` for every
    named event.
  * ``JobCreated.data[0]`` decoded as an address falls inside the known
    evaluator set for 3,876 of 3,876 non-zero samples.
  * ``PaymentReleased.data`` is 90,000 against a ``JobFunded`` amount of
    100,000 - the payout net of the 10% fee - and its ``topic[2]`` equals the
    provider from ``JobCreated``, which is what distinguishes it from
    ``JobCompleted`` (both occur exactly 14,644 times).
  * ``JobCompleted.data`` is a 32-byte deliverable hash, not an amount.

The four signatures the bundled scanner reports as "UNMAPPED ... cancel /
reject / expire lives here" are EvaluatorFeePaid, Refunded, JobExpired and
JobRejected, identified by count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol

# --------------------------------------------------------------------------
# Event signatures (topic[0])
# --------------------------------------------------------------------------

JOB_CREATED = "0xb0f0239bfdd96453e24733e18bfc24b70d8fadf123dd977473518dd577ee79b9"
BUDGET_SET = "0x869e2577b006bf47ee981cf6fec2e25583548081c14b98deab587f77b5068038"
JOB_FUNDED = "0xe3fbcc1ea1bdc559ec7f0347efde7655e58b5f45a30b0e4470a583c3ef5496b3"
JOB_SUBMITTED = "0x80c17db79857f338a6a6df68a6883ecc0ce78e2202fe61ed979733573f40538e"
JOB_COMPLETED = "0x0fd54bd364fa9e67f17b091aefe930932c09fe7651cf5ad02c71a418f3341444"
PAYMENT_RELEASED = "0x21d71db5be59bb9fa133895586b7404307dd33fb93b16db09dc6f1d9d7d231b0"
EVALUATOR_FEE_PAID = "0x253dd534010ac976fa263caa123bae79b9c50292adf7ce67bdc5ec309f784e61"
REFUNDED = "0x7ca5472b7ea78c2c0141c5a12ee6d170cf4ce8ed06be3d22c8252ddfc7a6a2c4"
JOB_EXPIRED = "0x97237956f8810192811e2c3f273fd02c5d6295206fdd9c62e6fe2bfc19ba9232"
JOB_REJECTED = "0xae7362b1af91f4492868987b9c73990d780060811551b58728fbe96fd1bab275"

SIGNATURES: dict[str, str] = {
    JOB_CREATED: "JobCreated",
    BUDGET_SET: "BudgetSet",
    JOB_FUNDED: "JobFunded",
    JOB_SUBMITTED: "JobSubmitted",
    JOB_COMPLETED: "JobCompleted",
    PAYMENT_RELEASED: "PaymentReleased",
    EVALUATOR_FEE_PAID: "EvaluatorFeePaid",
    REFUNDED: "Refunded",
    JOB_EXPIRED: "JobExpired",
    JOB_REJECTED: "JobRejected",
}

#: Events that terminate a funded job. See spec 10.2 - these are explanatory
#: fields on a failure episode, never competing labels, and they never change
#: the prediction target.
TERMINAL_FAILURE = frozenset({"JobRejected", "Refunded", "JobExpired"})

USDC_DECIMALS = 6


def usdc(raw: int) -> float:
    """Raw on-chain amount to USDC."""
    return raw / 10**USDC_DECIMALS


# --------------------------------------------------------------------------
# Decoded event
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AcpEvent:
    """One decoded ACP log.

    ``block`` and ``log_index`` are the ordering key. Time is always block
    number, never wall clock (spec 10.3).
    """

    kind: str
    job_id: int
    block: int
    log_index: int
    timestamp: int | None = None
    fields: dict[str, object] = field(default_factory=dict)

    @property
    def order_key(self) -> tuple[int, int, int]:
        """Stable total order: (block, log_index, job_id) per spec 10.3."""
        return (self.block, self.log_index, self.job_id)


def _words(data: str) -> list[str]:
    body = data[2:] if data.startswith("0x") else data
    return [body[i : i + 64] for i in range(0, len(body), 64)]


def _addr(word: str) -> str:
    """Last 20 bytes of a 32-byte word, as a lowercase 0x address.

    Accepts both forms this data contains: topics arrive 0x-prefixed, while
    ``_words`` yields bare 64-char words. Normalising here rather than at each
    call site avoids the off-by-two that leaves a ``00`` glued to the front of
    the address and silently splits one agent into two identities.
    """
    body = word[2:] if word.startswith("0x") else word
    return "0x" + body[-40:].lower()


def decode_log(log: dict) -> AcpEvent | None:
    """Decode one raw log. Returns None for signatures we do not model."""
    topics = log.get("topics") or []
    if not topics:
        return None
    kind = SIGNATURES.get(topics[0].lower())
    if kind is None:
        return None

    # Every modelled event indexes jobId as topic[1]. JobExpired and BudgetSet
    # carry nothing else indexed.
    if len(topics) < 2:
        return None

    job_id = int(topics[1], 16)
    block = int(log["blockNumber"], 16)
    log_index = int(log["logIndex"], 16)
    ts_raw = log.get("blockTimestamp")
    timestamp = int(ts_raw, 16) if ts_raw else None
    data = _words(log.get("data") or "0x")

    f: dict[str, object] = {}

    if kind == "JobCreated":
        # JobCreated(jobId, client, provider, evaluator, expiredAt, ...)
        f["client"] = _addr(topics[2])
        f["provider"] = _addr(topics[3])
        if data:
            f["evaluator"] = _addr(data[0])
            if int(data[0], 16) == 0:
                f["evaluator"] = None
        if len(data) > 1:
            f["expired_at"] = int(data[1], 16)

    elif kind == "BudgetSet":
        f["amount"] = int(data[0], 16) if data else 0

    elif kind == "JobFunded":
        f["funder"] = _addr(topics[2])
        f["amount"] = int(data[0], 16) if data else 0

    elif kind == "JobCompleted":
        # topic[2] is the evaluator; data is the deliverable hash, not an
        # amount. The hash is deliberately unused - task quality is not
        # observable, so no feature may depend on it (spec 3).
        f["evaluator"] = _addr(topics[2])

    elif kind == "PaymentReleased":
        f["payee"] = _addr(topics[2])
        f["amount"] = int(data[0], 16) if data else 0

    elif kind == "JobRejected":
        f["rejector"] = _addr(topics[2])

    elif kind in ("Refunded", "EvaluatorFeePaid"):
        f["recipient"] = _addr(topics[2])
        f["amount"] = int(data[0], 16) if data else 0

    # JobExpired carries only the job id.

    return AcpEvent(
        kind=kind,
        job_id=job_id,
        block=block,
        log_index=log_index,
        timestamp=timestamp,
        fields=f,
    )


# --------------------------------------------------------------------------
# Event sources
# --------------------------------------------------------------------------


class EventSource(Protocol):
    """Anything that can yield ACP events in chronological order.

    Replay and live operation differ only in which source is plugged in. The
    prediction pipeline downstream cannot tell them apart, which is what keeps
    a single code path (and so a single determinism guarantee) across both.
    """

    def iter_events(self) -> Iterator[AcpEvent]: ...


class LocalDatasetSource:
    """Events from a scanner state file on disk.

    This is the source of truth for backtesting and reproducibility (spec
    10.6). It is never copied wholesale into memory.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_events(self) -> Iterator[AcpEvent]:
        with self.path.open() as fh:
            state = json.load(fh)
        yield from sort_events(
            ev for ev in map(decode_log, state["logs"]) if ev is not None
        )

    def cursor(self) -> int:
        """Block the scanner reached, for handing off to a live source."""
        with self.path.open() as fh:
            return int(json.load(fh)["state"]["cursor"])


def sort_events(events: Iterable[AcpEvent]) -> list[AcpEvent]:
    """Total order by (block, log_index, job_id).

    Ties broken deterministically so that two runs over the same data produce
    identical decisions (spec 10.3).
    """
    return sorted(events, key=lambda e: e.order_key)
