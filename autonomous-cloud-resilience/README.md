# Autonomous Cloud Resilience Framework
### Using ML + Consensus Algorithms + Q-Learning

---

## Project Structure

```
autonomous-cloud-resilience/
├── config/
│   ├── aws_config.json        # SQS/SNS URLs + region
│   ├── nodes.json             # Node IDs + EC2 instance IDs
│   └── runtime_config.json   # Intervals, thresholds, log limits
├── data/
│   ├── metrics_logged.jsonl  # Live dataset (max 3000 rows)
│   └── metrics_labeled.csv  # CSV fallback / synthetic data
├── models/
│   ├── best_model.pkl        # Saved best ML model
│   ├── model_report.json     # F1/accuracy for all 3 models
│   └── q_table.npy           # Q-Learning table
├── common/                   # Shared types, logger, utils
├── aws/                      # SQS client (real + mock)
├── node/                     # Node Agent (runs on each EC2)
├── controller/               # Central controller
├── ml/                       # Dataset + 3 ML models
└── rl/                       # Q-Learning agent + policy
```

---

## Quick Start (Local / Offline)

> **No AWS required.** `USE_MOCK_SQS=1` (default) uses an in-memory queue.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the controller (trains models, then starts live loop)
python -m controller.controller

# 3. In separate terminals, run node agents
python -m node.node_agent --node-id N1
python -m node.node_agent --node-id N2
python -m node.node_agent --node-id N3
```

---

## Running on AWS

1. Edit `config/aws_config.json` with your real SQS queue URLs.
2. Set `USE_MOCK_SQS=0` in your environment.
3. Deploy `node_agent.py` on each EC2 / Docker container.
4. Run `controller.py` on a central instance.

---

## System States

| State | Value | Meaning |
|-------|-------|---------|
| NORMAL   | 0 | Healthy — round-robin scheduling |
| WARNING  | 1 | Congestion likely — priority + RL scheduling |
| CRITICAL | 2 | Failure risk — isolate, re-elect leader, migrate tasks |

---

## ML Models Compared

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Supervised | Baseline |
| Random Forest       | Supervised | Main candidate |
| Isolation Forest    | Unsupervised | Anomaly detection on NORMAL samples |

Best model selected by **F1-weighted** score, saved to `models/best_model.pkl`.

---

## Q-Learning Scheduler

- **State**: (system_state × cpu_bucket × queue_bucket)  
- **Actions**: MORE_CPU_TO_CRITICAL / INCREASE_TIME_SLICE / DEFER_NON_CRITICAL  
- **Reward**: improvement in average latency + queue length  
- Q-table persisted to `models/q_table.npy`

---

## Configuration

`config/runtime_config.json` controls all timing and thresholds:

```json
{
  "heartbeat_interval_sec": 2,
  "control_interval_sec":  30,
  "log_interval_sec":       5,
  "max_log_rows":        3000,
  "heartbeat_timeout_sec":  10,
  "mid_cpu_threshold":    70.0,
  "mid_queue_threshold":     5,
  "high_lat_threshold":  300.0
}
```
