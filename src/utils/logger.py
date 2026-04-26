"""
logger.py — JSON Timestamped Agent Action Logger

Logs every agent action with timestamps to a JSONL file.
Satisfies the project requirement: "Every agent action logged with timestamps (JSON)."

Team: Benmouma Salma, Gassi Oumaima
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class AgentLogger:
    """
    Structured JSON logger for multi-agent system actions.
    
    Each log entry contains:
        - timestamp (ISO 8601)
        - agent name
        - action performed
        - input data
        - output data
        - status (success / error)
        - duration in milliseconds
        - optional error message
    
    Logs are appended to a JSONL file (one JSON object per line).
    """

    def __init__(self, log_dir: str = "logs", log_file: str = "agent_actions.jsonl"):
        """
        Initialize the agent logger.

        Args:
            log_dir: Directory to store log files.
            log_file: Name of the JSONL log file.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / log_file

        # Also set up standard Python logging for console output
        self._setup_console_logger()

    def _setup_console_logger(self):
        """Configure standard Python logger for console output."""
        self.console_logger = logging.getLogger("SmartDocAnalyst")
        if not self.console_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.console_logger.addHandler(handler)
            self.console_logger.setLevel(logging.INFO)

    def log_action(
        self,
        agent: str,
        action: str,
        input_data: Any = None,
        output_data: Any = None,
        status: str = "success",
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Log a single agent action to the JSONL file.

        Args:
            agent: Name of the agent performing the action.
            action: Name of the action/tool being used.
            input_data: Input provided to the action.
            output_data: Output returned by the action.
            status: Either "success" or "error".
            duration_ms: Time taken in milliseconds.
            error: Error message if status is "error".
            metadata: Additional metadata to include.

        Returns:
            The log entry dictionary that was written.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "action": action,
            "input": self._safe_serialize(input_data),
            "output": self._safe_serialize(output_data),
            "status": status,
            "duration_ms": round(duration_ms, 2) if duration_ms else None,
        }

        if error:
            entry["error"] = str(error)

        if metadata:
            entry["metadata"] = metadata

        # Write to JSONL file
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            self.console_logger.error(f"Failed to write log entry: {e}")

        # Also log to console
        log_msg = f"[{agent}] {action} → {status}"
        if duration_ms:
            log_msg += f" ({duration_ms:.0f}ms)"
        if status == "error":
            self.console_logger.error(log_msg)
        else:
            self.console_logger.info(log_msg)

        return entry

    def log_pipeline_start(self, input_file: str):
        """Log the start of a document analysis pipeline."""
        self.log_action(
            agent="Orchestrator",
            action="pipeline_start",
            input_data={"file": input_file},
            status="success"
        )
        self.console_logger.info(f"{'='*60}")
        self.console_logger.info(f"Pipeline started for: {input_file}")
        self.console_logger.info(f"{'='*60}")

    def log_pipeline_end(self, total_duration_ms: float, success: bool = True):
        """Log the end of a document analysis pipeline."""
        self.log_action(
            agent="Orchestrator",
            action="pipeline_end",
            output_data={"success": success},
            status="success" if success else "error",
            duration_ms=total_duration_ms
        )
        self.console_logger.info(f"{'='*60}")
        self.console_logger.info(
            f"Pipeline {'completed successfully' if success else 'failed'} "
            f"in {total_duration_ms:.0f}ms"
        )
        self.console_logger.info(f"{'='*60}")

    def get_logs(self) -> list[dict]:
        """Read and return all log entries from the JSONL file."""
        logs = []
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return logs

    @staticmethod
    def _safe_serialize(data: Any) -> Any:
        """
        Safely serialize data for JSON output.
        Truncates very long strings to avoid bloated logs.
        """
        if data is None:
            return None
        if isinstance(data, str):
            # Truncate very long text (e.g., full document content)
            if len(data) > 1000:
                return data[:1000] + f"... [truncated, total {len(data)} chars]"
            return data
        if isinstance(data, dict):
            return {k: AgentLogger._safe_serialize(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [AgentLogger._safe_serialize(item) for item in data]
        if isinstance(data, (int, float, bool)):
            return data
        return str(data)
