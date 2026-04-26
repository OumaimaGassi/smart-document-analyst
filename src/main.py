"""
main.py — Entry Point for Smart Document Analyst

Run the multi-agent document analysis pipeline from the command line.

Usage:
    python src/main.py --input path/to/document.pdf
    python src/main.py --input path/to/document.pdf --hitl
    python src/main.py --input path/to/document.pdf --verbose

Team: Benmouma Salma, Gassi Oumaima
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.crew import SmartDocumentAnalystCrew
from src.utils.logger import AgentLogger


def main():
    """Main entry point for the Smart Document Analyst."""

    parser = argparse.ArgumentParser(
        description="Smart Document Analyst — Multi-Agent AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --input data/sample_docs/invoice.pdf
  python src/main.py --input document.png --hitl --verbose
  python src/main.py --input report.pdf --model model/document_classifier.pt
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the document file to analyze (PDF, PNG, JPG, TIFF)"
    )
    parser.add_argument(
        "--model", "-m",
        default="model/document_classifier.pt",
        help="Path to the trained CNN model (default: model/document_classifier.pt)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs/reports",
        help="Directory for generated reports (default: outputs/reports)"
    )
    parser.add_argument(
        "--hitl",
        action="store_true",
        help="Enable Human-in-the-Loop checkpoint for classification approval"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose agent output"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (skip model loading, use mock data)"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY not set. LLM summarization will use fallback mode.")
        print("   Set it in .env or as environment variable for full functionality.\n")

    # Banner
    print("=" * 60)
    print("📄 Smart Document Analyst — Multi-Agent AI System")
    print("   Team: Benmouma Salma, Gassi Oumaima")
    print("   UIR S8 — AI & Big Data 2025–2026")
    print("=" * 60)
    print(f"\n📂 Input:  {args.input}")
    print(f"🧠 Model:  {args.model}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🧑 HITL:   {'Enabled' if args.hitl else 'Disabled'}")
    print(f"🔊 Verbose: {'Yes' if args.verbose else 'No'}")
    print()

    # Initialize logger
    logger = AgentLogger()

    # Create and run the crew
    crew = SmartDocumentAnalystCrew(
        model_path=args.model,
        output_dir=args.output_dir,
        verbose=args.verbose,
        logger=logger
    )

    if args.hitl:
        result = crew.run_with_hitl(args.input)
    else:
        result = crew.run(args.input)

    # Display results
    print("\n" + "=" * 60)
    print("📊 PIPELINE RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") == "success":
        print("\n✅ Document analysis completed successfully!")
    elif result.get("status") == "rejected":
        print("\n⚠️  Document was rejected at HITL checkpoint.")
    else:
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")

    return result


if __name__ == "__main__":
    main()
