"""
test_agents.py — Unit Tests for Agent Definitions

Tests that agents are properly configured with correct roles,
tools, and settings.

Team: Benmouma Salma, Gassi Oumaima
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAgentCreation:
    """Test that agents can be created with proper configuration."""

    @patch("src.tools.cnn_classify_tool.CNNClassifyTool._load_model")
    def test_classifier_agent_creation(self, mock_load):
        """Classifier agent should have the CNN tool."""
        mock_load.return_value = None
        from src.agents.classifier_agent import create_classifier_agent

        agent = create_classifier_agent(
            model_path="fake.pt", logger=MagicMock(), verbose=False
        )
        assert agent.role == "Document Classifier"
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "cnn_classify_tool"

    def test_extractor_agent_creation(self):
        """Extractor agent should have OCR and summarization tools."""
        from src.agents.extractor_agent import create_extractor_agent

        agent = create_extractor_agent(logger=MagicMock(), verbose=False)
        assert agent.role == "Text Extractor and Summarizer"
        assert len(agent.tools) == 2
        tool_names = [t.name for t in agent.tools]
        assert "ocr_extract_tool" in tool_names
        assert "llm_summarize_tool" in tool_names

    def test_reporter_agent_creation(self):
        """Reporter agent should have the report builder tool."""
        from src.agents.reporter_agent import create_reporter_agent

        agent = create_reporter_agent(logger=MagicMock(), verbose=False)
        assert agent.role == "Report Generator"
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "report_builder_tool"


class TestLoggerIntegration:
    """Test that the logger works correctly."""

    def test_log_action(self):
        """Logger should write actions to JSONL file."""
        import json
        from src.utils.logger import AgentLogger

        logger = AgentLogger(log_dir="test_logs", log_file="test.jsonl")
        logger.log_action(
            agent="TestAgent",
            action="test_action",
            input_data={"key": "value"},
            output_data={"result": "ok"},
            status="success",
            duration_ms=123.45
        )

        logs = logger.get_logs()
        assert len(logs) >= 1
        last_log = logs[-1]
        assert last_log["agent"] == "TestAgent"
        assert last_log["status"] == "success"

        # Cleanup
        import shutil
        if Path("test_logs").exists():
            shutil.rmtree("test_logs")

    def test_hitl_checkpoint(self):
        """HITL checkpoint should work with mocked input."""
        from src.utils.hitl import HumanInTheLoop

        hitl = HumanInTheLoop(logger=MagicMock())

        with patch("builtins.input", return_value="yes"):
            result = hitl.classification_checkpoint(
                predicted_class="invoice",
                confidence=0.95,
                top_3=[
                    {"class": "invoice", "confidence": 0.95},
                    {"class": "form", "confidence": 0.03},
                    {"class": "letter", "confidence": 0.02}
                ],
                file_path="test.pdf"
            )

        assert result["approved"] is True
        assert result["final_class"] == "invoice"
        assert result["was_overridden"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
