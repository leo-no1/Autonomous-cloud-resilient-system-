"""
aws/sqs_client.py
Thin wrapper around boto3 SQS with a local in-memory mock for
offline / unit-test use.  Set env var USE_MOCK_SQS=1 to use the mock.
"""

import json
import os
import queue
import time
from typing import Any, Dict, List, Optional

from common.logger import get_logger

log = get_logger(__name__)

# ── Mock (local) SQS ─────────────────────────────────────────────────────────

_mock_queues: Dict[str, queue.Queue] = {}


def _get_mock_queue(url: str) -> queue.Queue:
    if url not in _mock_queues:
        _mock_queues[url] = queue.Queue()
    return _mock_queues[url]


class MockSQSMessage:
    def __init__(self, body: str):
        self.body           = body
        self.receipt_handle = str(id(self))


# ── Real SQS ──────────────────────────────────────────────────────────────────

def _boto_client():
    import boto3
    from common.utils import load_json
    cfg = load_json("config/aws_config.json")
    return boto3.client("sqs", region_name=cfg.get("region", "us-east-1"))


# ── Public API ────────────────────────────────────────────────────────────────

USE_MOCK = os.environ.get("USE_MOCK_SQS", "1") == "1"


def sqs_send(queue_url: str, message_dict: Dict[str, Any]) -> None:
    body = json.dumps(message_dict)
    if USE_MOCK:
        _get_mock_queue(queue_url).put(MockSQSMessage(body))
        log.debug("MOCK SQS SEND -> %s | %s", queue_url, body[:120])
    else:
        client = _boto_client()
        client.send_message(QueueUrl=queue_url, MessageBody=body)
        log.debug("SQS SEND -> %s", queue_url)


def sqs_poll(
    queue_url: str,
    max_msgs: int = 10,
    wait_time: int = 2,
) -> List[Dict[str, Any]]:
    """Returns list of dicts: {'body': dict, 'receipt_handle': str}"""
    if USE_MOCK:
        q = _get_mock_queue(queue_url)
        results = []
        deadline = time.time() + wait_time
        while len(results) < max_msgs and time.time() < deadline:
            try:
                msg = q.get_nowait()
                results.append({
                    "body": json.loads(msg.body),
                    "receipt_handle": msg.receipt_handle,
                })
            except queue.Empty:
                time.sleep(0.05)
        return results
    else:
        client = _boto_client()
        resp = client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=min(max_msgs, 10),
            WaitTimeSeconds=wait_time,
        )
        out = []
        for m in resp.get("Messages", []):
            out.append({
                "body": json.loads(m["Body"]),
                "receipt_handle": m["ReceiptHandle"],
            })
        return out


def sqs_delete(queue_url: str, receipt_handle: str) -> None:
    if USE_MOCK:
        return  # mock queue items are auto-consumed
    client = _boto_client()
    client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
