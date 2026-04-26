"""
crew.py — CrewAI Crew Definition

Defines the Smart Document Analyst crew with 3 specialist agents
and a sequential orchestration process.

Team: Benmouma Salma, Gassi Oumaima
"""

import json
from typing import Optional

from crewai import Crew, Task, Process

from src.agents.classifier_agent import create_classifier_agent
from src.agents.extractor_agent import create_extractor_agent
from src.agents.reporter_agent import create_reporter_agent
from src.utils.logger import AgentLogger
from src.utils.hitl import HumanInTheLoop


class SmartDocumentAnalystCrew:
    """
    Multi-agent crew for document analysis.
    
    Pipeline:
        1. Classifier Agent → CNN classification
        2. HITL Checkpoint → Human approval
        3. Extractor Agent → OCR + summarization
        4. Reporter Agent → Report generation
    """

    def __init__(
        self,
        model_path: str = "model/document_classifier.pt",
        output_dir: str = "outputs/reports",
        verbose: bool = True,
        logger: Optional[AgentLogger] = None
    ):
        """
        Initialize the Smart Document Analyst crew.

        Args:
            model_path: Path to the trained CNN model.
            output_dir: Directory for generated reports.
            verbose: Enable verbose agent output.
            logger: AgentLogger instance.
        """
        self.logger = logger or AgentLogger()
        self.hitl = HumanInTheLoop(logger=self.logger)
        self.verbose = verbose

        # Create specialist agents
        self.classifier_agent = create_classifier_agent(
            model_path=model_path, logger=self.logger, verbose=verbose
        )
        self.extractor_agent = create_extractor_agent(
            logger=self.logger, verbose=verbose
        )
        self.reporter_agent = create_reporter_agent(
            output_dir=output_dir, logger=self.logger, verbose=verbose
        )

    def create_tasks(self, file_path: str) -> list[Task]:
        """
        Create the sequential task pipeline for document analysis.

        Args:
            file_path: Path to the document to analyze.

        Returns:
            List of CrewAI Task objects.
        """
        # Task 1: Classify the document
        classify_task = Task(
            description=(
                f"Classify the document at '{file_path}' using the CNN classification tool. "
                f"Call the cnn_classify_tool with the file path and return the full "
                f"classification result including the predicted class, confidence score, "
                f"and top 3 predictions."
            ),
            expected_output=(
                "A JSON object containing: 'class' (predicted document type), "
                "'confidence' (float between 0 and 1), 'top_3' (list of top 3 predictions "
                "with class and confidence), and 'file_path'."
            ),
            agent=self.classifier_agent,
        )

        # Task 2: Extract text and summarize
        extract_task = Task(
            description=(
                f"Extract all text from the document at '{file_path}' using the OCR extraction tool. "
                f"Then summarize the extracted text using the LLM summarization tool. "
                f"Use the document type from the previous classification result to adapt "
                f"the summarization style. Return both the extracted text and the summary."
            ),
            expected_output=(
                "Two JSON results: (1) extraction result with 'text', 'num_pages', 'method'; "
                "(2) summary result with 'summary', 'key_points', 'word_count'."
            ),
            agent=self.extractor_agent,
            context=[classify_task],
        )

        # Task 3: Generate the report
        report_task = Task(
            description=(
                f"Generate a professional analysis report for the document at '{file_path}'. "
                f"Use the report_builder_tool with the classification results, extraction results, "
                f"and summary results from the previous tasks. Pass these as JSON strings "
                f"to the tool's 'classification', 'extraction', and 'summary' parameters."
            ),
            expected_output=(
                "A JSON object with 'report_path' (path to the generated report file), "
                "'format' (markdown/pdf), and 'report_length'."
            ),
            agent=self.reporter_agent,
            context=[classify_task, extract_task],
        )

        return [classify_task, extract_task, report_task]

    def run(self, file_path: str) -> dict:
        """
        Run the full document analysis pipeline.

        Args:
            file_path: Path to the document to analyze.

        Returns:
            Dictionary with the final pipeline results.
        """
        import time
        start_time = time.time()

        self.logger.log_pipeline_start(file_path)

        # Create tasks
        tasks = self.create_tasks(file_path)

        # Create and run the crew
        crew = Crew(
            agents=[self.classifier_agent, self.extractor_agent, self.reporter_agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=self.verbose,
        )

        try:
            result = crew.kickoff()

            total_duration = (time.time() - start_time) * 1000
            self.logger.log_pipeline_end(total_duration, success=True)

            return {
                "status": "success",
                "result": str(result),
                "duration_ms": total_duration
            }

        except Exception as e:
            total_duration = (time.time() - start_time) * 1000
            self.logger.log_pipeline_end(total_duration, success=False)

            return {
                "status": "error",
                "error": str(e),
                "duration_ms": total_duration
            }

    def run_with_hitl(self, file_path: str) -> dict:
        """
        Run the pipeline with the Human-in-the-Loop checkpoint.
        
        This manually orchestrates the pipeline to insert the HITL
        checkpoint between classification and extraction.

        Args:
            file_path: Path to the document to analyze.

        Returns:
            Dictionary with the final pipeline results.
        """
        import time
        start_time = time.time()

        self.logger.log_pipeline_start(file_path)

        try:
            # Step 1: Classification
            print("\n🔍 Step 1/4: Classifying document...")
            classify_task = Task(
                description=(
                    f"Classify the document at '{file_path}' using the cnn_classify_tool. "
                    f"Return the complete JSON result."
                ),
                expected_output="JSON with class, confidence, top_3, file_path",
                agent=self.classifier_agent,
            )

            classify_crew = Crew(
                agents=[self.classifier_agent],
                tasks=[classify_task],
                process=Process.sequential,
                verbose=self.verbose,
            )

            classify_result = classify_crew.kickoff()
            classify_data = json.loads(str(classify_result))

            # Step 2: HITL Checkpoint
            print("\n🧑 Step 2/4: Human-in-the-Loop checkpoint...")
            hitl_result = self.hitl.classification_checkpoint(
                predicted_class=classify_data.get("class", "unknown"),
                confidence=classify_data.get("confidence", 0),
                top_3=classify_data.get("top_3", []),
                file_path=file_path
            )

            if hitl_result["rejected"]:
                total_duration = (time.time() - start_time) * 1000
                self.logger.log_pipeline_end(total_duration, success=False)
                return {
                    "status": "rejected",
                    "message": "Document rejected by user at HITL checkpoint",
                    "duration_ms": total_duration
                }

            # Update classification with HITL decision
            final_class = hitl_result["final_class"]
            if hitl_result["was_overridden"]:
                classify_data["class"] = final_class
                classify_data["hitl_overridden"] = True

            # Step 3: Extraction + Summarization
            print("\n📝 Step 3/4: Extracting and summarizing...")
            extract_task = Task(
                description=(
                    f"Extract text from '{file_path}' using ocr_extract_tool, then "
                    f"summarize it using llm_summarize_tool with doc_type='{final_class}'. "
                    f"Return both extraction and summary results."
                ),
                expected_output="Extraction and summary JSON results",
                agent=self.extractor_agent,
            )

            extract_crew = Crew(
                agents=[self.extractor_agent],
                tasks=[extract_task],
                process=Process.sequential,
                verbose=self.verbose,
            )
            extract_result = extract_crew.kickoff()

            # Step 4: Report Generation
            print("\n📊 Step 4/4: Generating report...")
            report_task = Task(
                description=(
                    f"Generate a report using report_builder_tool. "
                    f"Classification: {json.dumps(classify_data)}. "
                    f"Extraction and summary: {str(extract_result)}. "
                    f"File path: {file_path}."
                ),
                expected_output="JSON with report_path and format",
                agent=self.reporter_agent,
            )

            report_crew = Crew(
                agents=[self.reporter_agent],
                tasks=[report_task],
                process=Process.sequential,
                verbose=self.verbose,
            )
            report_result = report_crew.kickoff()

            total_duration = (time.time() - start_time) * 1000
            self.logger.log_pipeline_end(total_duration, success=True)

            return {
                "status": "success",
                "classification": classify_data,
                "hitl": hitl_result,
                "report": str(report_result),
                "duration_ms": total_duration
            }

        except Exception as e:
            total_duration = (time.time() - start_time) * 1000
            self.logger.log_pipeline_end(total_duration, success=False)
            return {
                "status": "error",
                "error": str(e),
                "duration_ms": total_duration
            }
