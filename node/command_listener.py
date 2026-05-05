"""
node/command_listener.py
Polls the SQS command queue and applies commands to the node state.
"""

from common.logger import get_logger
from common.types import CommandMessage, CommandType, NodeRole
from common.utils import load_json
from aws.sqs_client import sqs_poll, sqs_delete

log = get_logger(__name__)


class CommandListener:
    def __init__(self, node_id: str, queue_url: str):
        self.node_id   = node_id
        self.queue_url = queue_url

    def poll_and_apply(self, node_state: dict) -> None:
        """
        node_state keys mutated in-place:
            role     : NodeRole
            isolated : bool
            shutdown : bool
        """
        messages = sqs_poll(self.queue_url, max_msgs=10, wait_time=2)
        for item in messages:
            body   = item["body"]
            handle = item["receipt_handle"]
            cmd    = CommandMessage.from_dict(body)

            if cmd.target_node_id != self.node_id:
                # Not for us — leave it (SQS visibility timeout will restore it)
                continue

            log.info("Received command: %s", cmd.command)

            if cmd.command == CommandType.ISOLATE:
                node_state["isolated"] = True
                log.warning("Node %s ISOLATED.", self.node_id)

            elif cmd.command == CommandType.SET_ROLE_LEADER:
                node_state["role"] = NodeRole.LEADER
                log.info("Node %s promoted to LEADER.", self.node_id)

            elif cmd.command == CommandType.SET_ROLE_FOLLOWER:
                node_state["role"] = NodeRole.FOLLOWER
                log.info("Node %s set to FOLLOWER.", self.node_id)

            elif cmd.command == CommandType.SHUTDOWN:
                node_state["shutdown"] = True
                log.warning("Node %s received SHUTDOWN.", self.node_id)

            sqs_delete(self.queue_url, handle)
