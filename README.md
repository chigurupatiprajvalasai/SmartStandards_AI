# SmartStandards AI — SIH26108 Prototype

AI-Powered Recommendation Engine for Identifying Applicable Indian Standards for Procurement Specifications.

## Quick Start

```bash
cd SmartStandards_AI_SIH26108
python -m pip install -r requirements.txt
streamlit run app.py
```

The application is designed for an offline/demo-first hackathon environment.

## Included prototype capabilities

- Product specification analyzer
- PDF/DOCX tender upload and text extraction
- Requirement extraction using rules + lightweight NLP
- Hybrid retrieval: TF-IDF/BM25-like lexical retrieval + semantic embeddings when available
- Offline semantic fallback using TF-IDF
- Ranked applicability score (0–100)
- Explainable requirement → standard evidence
- Curated standards knowledge graph
- Normative/allied/supersession relationships
- Current/superseded/withdrawn lifecycle indicators
- Mandatory/voluntary metadata
- Compliance gap analysis
- Contradiction/duplicate requirement checks
- Procurement specification generator
- Compliance checklist generator
- PDF report generation
- CSV compliance export
- English/Hindi query normalization (demo translation dictionary; extensible)
- Reviewer accept/reject/flag workflow
- Audit trail
- Searchable standards library
- Interactive relationship graph
- Offline demo mode

## Important data note

This is a hackathon prototype using a small curated representative dataset. It does NOT claim to contain the complete BIS catalogue. Standard identifiers/titles and relationships in the demo dataset should be verified against authoritative BIS publications before real procurement use.

## Demo scenarios

Try:
1. "Procure flame-retardant PVC insulated copper cables for building installations, 1.1 kV."
2. "Procurement of safety helmets for industrial workers."
3. "Need drinking water storage tanks for institutional buildings."
4. Upload the included `sample_data/sample_tender.txt` (or create a PDF/DOCX from it).

## Architecture

Streamlit UI -> local document parser -> requirement extractor -> hybrid retrieval -> knowledge graph -> applicability scorer -> compliance engine -> report generator.

Optional packages:
- sentence-transformers: enables local embedding retrieval if installed.
- networkx: graph visualization.
- pypdf/python-docx: document extraction.
