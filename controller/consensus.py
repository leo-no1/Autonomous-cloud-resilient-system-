"""
controller/consensus.py
Simplified Raft-like leader election.
The controller runs this — it decides who the leader is and
sends SET_ROLE_LEADER / SET_ROLE_FOLLOWER commands via SQS.
"""

from typing import List, Optional

from common.logger import get_logger
from common.types import Node, NodeRole, NodeStatus, CommandType
from common.utils import majority
from aws.sqs_client import sqs_send

log = get_logger(__name__)


def _vote_granted(candidate: Node) -> bool:
    """Simplified: vote YES if candidate is HEALTHY."""
    return candidate.status == NodeStatus.HEALTHY


def elect_leader_if_needed(
    nodes: List[Node],
    current_leader_id: Optional[str],
    cmd_queue_url: str,
) -> Optional[str]:
    """
    Run a leader election round.
    Returns the elected leader's node_id, or None if election fails.
    """
    healthy = [n for n in nodes if n.status == NodeStatus.HEALTHY]

    if not healthy:
        log.error("No healthy nodes — cannot elect leader.")
        return current_leader_id

    # Pick candidate: lowest id (stable, deterministic)
    candidate = min(healthy, key=lambda n: n.id)

    votes = sum(1 for voter in healthy if _vote_granted(candidate))

    if votes >= majority(len(healthy)):
        log.info(
            "Leader elected: %s  (votes=%d/%d)",
            candidate.id, votes, len(healthy),
        )
        for n in healthy:
            cmd = CommandType.SET_ROLE_LEADER if n.id == candidate.id \
                  else CommandType.SET_ROLE_FOLLOWER
            n.role = NodeRole.LEADER if cmd == CommandType.SET_ROLE_LEADER \
                     else NodeRole.FOLLOWER
            sqs_send(cmd_queue_url, {
                "msg_type":       "COMMAND",
                "target_node_id": n.id,
                "command":        cmd.value,
            })
        return candidate.id

    log.warning("Election failed (votes=%d, needed=%d). Retry next cycle.",
                votes, majority(len(healthy)))
    return current_leader_id
