# Smart Document Analyst — Multi-Agent AI System

> **UIR S8 — Integrated Project: Building Multi-Agent AI Systems**
> AI & Big Data Program | Prof. Hakim Hafidi | 2025–2026

**Team:** Benmouma Salma, Gassi Oumaima

---

## Project Overview

A multi-agent system where specialized AI agents collaborate to **classify**, **extract & summarize**, and **generate professional reports** from documents. The system uses a **CNN-based deep learning model** (fine-tuned ResNet-18) for document classification, integrated as a functional tool within a **CrewAI** orchestration framework.

### Key Features

- **CNN Document Classifier** — Fine-tuned ResNet-18 on RVL-CDIP dataset (PyTorch)
- **3 Specialized Agents** — Classifier, Extractor & Summarizer, Report Generator
- **CrewAI Orchestration** — Sequential pipeline with manager coordination
- **Human-in-the-Loop** — Classification approval checkpoint
- **JSON Logging** — Every agent action logged with timestamps
- **Error Handling** — Graceful failure recovery, no crashes

---

## Architecture

```
User Input (PDF/Image)
    │
    ▼
┌──────────────────────┐
│   Orchestrator Agent  │  ← Manages workflow
└──────────┬───────────┘
           │
    ┌──────▼──────┐
    │  Classifier  │  ← CNN Model Tool (PyTorch)
    │    Agent     │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   HITL      │  ← Human approves classification
    │  Checkpoint  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Extractor & │  ← OCR + LLM Summarization
    │  Summarizer  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Report    │  ← Markdown/PDF Generation
    │  Generator  │
    └──────┬──────┘
           │
           ▼
    Final Analysis Report
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/smart-document-analyst.git
cd smart-document-analyst
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Download/Train the Model

Option A: Use the pre-trained model (place `document_classifier.pt` in `model/`)

Option B: Train from scratch using the notebook:

```bash
jupyter notebook notebooks/training.ipynb
```

### 4. Run the System

```bash
python src/main.py --input path/to/document.pdf
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── agents.yaml
│   └── tasks.yaml
├── src/
│   ├── main.py
│   ├── crew.py
│   ├── agents/
│   │   ├── classifier_agent.py
│   │   ├── extractor_agent.py
│   │   └── reporter_agent.py
│   ├── tools/
│   │   ├── cnn_classify_tool.py
│   │   ├── ocr_extract_tool.py
│   │   ├── llm_summarize_tool.py
│   │   └── report_builder_tool.py
│   ├── models/
│   │   └── document_classifier.py
│   └── utils/
│       ├── logger.py
│       ├── hitl.py
│       └── preprocessing.py
├── model/
│   └── document_classifier.pt
├── notebooks/
│   └── training.ipynb
├── data/
│   └── sample_docs/
├── outputs/
│   └── reports/
├── logs/
└── tests/
    ├── test_tools.py
    └── test_agents.py
```

---

## Evaluation

| Metric              | Target                                |
| ------------------- | ------------------------------------- |
| CNN Accuracy        | ≥ 85% on test set                    |
| Agent Collaboration | Genuine sequential pipeline           |
| Error Handling      | No crashes on edge cases              |
| HITL                | Interactive classification approval   |
| Logging             | JSON with timestamps for every action |

---

## Deliverables

- [X] GitHub repository with full source code
- [X] Trained CNN model (`document_classifier.pt`)
- [ ] PDF report (8–12 pages)
- [ ] Demo video (3–5 min)
- [ ] Presentation slides

---

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [RVL-CDIP Dataset](https://huggingface.co/datasets/rvl_cdip)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Google Gemini API](https://ai.google.dev/)

---

*Built with ❤️ by Benmouma Salma & Gassi Oumaima — UIR 2025–2026*
