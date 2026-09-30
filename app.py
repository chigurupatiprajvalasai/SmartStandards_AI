import re
from pathlib import Path

import pandas as pd
import streamlit as st

from core.audit import load_history, record_action
from core.data import load_demo_tender, load_standards
from core.document import extract_uploaded_text
from core.graph import graph_figure, standard_graph
from core.pipeline import analyze_text
from core.reports import build_csv_report, build_pdf_report

st.set_page_config(
    page_title="SmartStandards AI",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent

# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
    }

    .hero p {
        margin: .45rem 0 0;
        opacity: .9;
        font-size: 1rem;
    }

    .card {
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1rem;
        background: #111827;
        min-height: 125px;
    }

    .score {
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .small {
        font-size: .86rem;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Load standards safely
# -------------------------------------------------------------------
try:
    standards = load_standards()
except Exception as exc:
    st.error("Unable to load the standards database.")
    st.exception(exc)
    st.stop()

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "accepted" not in st.session_state:
    st.session_state.accepted = {}

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.title("🇮🇳 SmartStandards AI")

    page = st.radio(
        "Navigate",
        [
            "Analyze",
            "Recommendations",
            "Compliance",
            "Specification Generator",
            "Standards Library",
            "Knowledge Graph",
            "History",
        ],
    )

    st.divider()
    st.caption("SIH26108 • Offline-first prototype")
    st.caption(f"Curated standards: {len(standards)}")

# -------------------------------------------------------------------
# Analyze
# -------------------------------------------------------------------
if page == "Analyze":
    st.markdown(
        """
        <div class="hero">
            <h1>SmartStandards AI</h1>
            <p>
                Evidence-grounded recommendation engine for
                Indian procurement specifications
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Analyze a procurement requirement")

    mode = st.radio(
        "Input mode",
        ["Free-text specification", "Tender PDF/DOCX", "Demo tender"],
        horizontal=True,
    )

    text = ""

    if mode == "Free-text specification":
        text = st.text_area(
            "Enter product / tender requirement",
            value=(
                "Procure flame-retardant PVC insulated copper electrical "
                "cables for building installations, 1.1 kV. Include "
                "applicable testing and safety requirements."
            ),
            height=190,
            placeholder=(
                "Example: Procure electrical cables for a building, "
                "with testing and safety requirements."
            ),
        )

    elif mode == "Demo tender":
        try:
            text = load_demo_tender()
            st.text_area("Demo tender", text, height=270)
        except Exception as exc:
            st.error("Unable to load demo tender.")
            st.exception(exc)

    else:
        upload = st.file_uploader(
            "Upload PDF, DOCX or TXT",
            type=["pdf", "docx", "txt"],
        )

        if upload:
            try:
                text = extract_uploaded_text(upload)
                st.text_area(
                    "Extracted document text",
                    text[:15000],
                    height=320,
                )

                if text.strip():
                    st.success("Document text extracted successfully.")
                else:
                    st.warning("The uploaded document contains no readable text.")
            except Exception as exc:
                st.error("Unable to extract document text.")
                st.exception(exc)

    if st.button(
        "🔎 Analyze & Recommend",
        type="primary",
        use_container_width=True,
    ):
        if not text or not text.strip():
            st.error("Please provide a specification or document.")
        else:
            try:
                with st.spinner("Analyzing procurement requirement..."):
                    result = analyze_text(
                        text,
                        standards,
                        language="English",
                    )

                st.session_state.analysis = result

                try:
                    record_action(
                        "analysis",
                        {
                            "query": text[:500],
                            "recommendations": [
                                r.get("standard_id", "")
                                for r in result.get("recommendations", [])
                            ],
                        },
                    )
                except Exception:
                    pass

                st.success("Analysis completed successfully.")
                st.rerun()

            except Exception as exc:
                st.error("Analysis failed.")
                st.exception(exc)

    # Results
    if st.session_state.analysis:
        result = st.session_state.analysis
        requirements = result.get("requirements", {})

        st.divider()
        st.subheader("Extracted requirements")

        def as_text(value):
            if isinstance(value, list):
                return ", ".join(str(x) for x in value) or "Not detected"
            return str(value) if value else "Not detected"

        cards = [
            ("Product", as_text(requirements.get("product"))),
            ("Materials", as_text(requirements.get("materials"))),
            ("Properties", as_text(requirements.get("properties"))),
            ("Application", as_text(requirements.get("application"))),
        ]

        cols = st.columns(4)
        for col, (label, value) in zip(cols, cards):
            col.markdown(
                f"""
                <div class="card">
                    <b>{label}</b><br><br>
                    <span class="score">{value}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Top recommendations")

        recommendations = result.get("recommendations", [])

        if not recommendations:
            st.warning("No applicable standards were found.")

        for rec in recommendations[:5]:
            sid = str(rec.get("standard_id", "Unknown"))
            title = str(rec.get("title", "Untitled standard"))
            scope = str(rec.get("scope", "No scope information available."))
            score = float(rec.get("score", 0) or 0)
            score = max(0, min(100, score))

            with st.container(border=True):
                left, right = st.columns([5, 1])

                left.markdown(f"### {sid} — {title}")
                left.write(scope)

                right.metric("Applicability", f"{score:.0f}/100")
                st.progress(score / 100)

                st.write("**Why this standard?**")
                reasons = rec.get("reasons", [])
                if reasons:
                    for reason in reasons:
                        st.write("✓ " + str(reason))
                else:
                    st.write("Semantic and metadata similarity identified this candidate.")

                evidence = rec.get("evidence", [])
                if evidence:
                    st.caption(
                        "Evidence: " + " | ".join(map(str, evidence))
                    )

                status = st.session_state.accepted.get(sid)
                if status:
                    st.info(f"Reviewer status: **{status}**")

                a, b, c = st.columns(3)

                if a.button("Accept", key=f"accept_{sid}"):
                    st.session_state.accepted[sid] = "Accepted"
                    try:
                        record_action("accept", {"standard_id": sid})
                    except Exception:
                        pass
                    st.rerun()

                if b.button("Flag", key=f"flag_{sid}"):
                    st.session_state.accepted[sid] = "Flagged"
                    try:
                        record_action("flag", {"standard_id": sid})
                    except Exception:
                        pass
                    st.rerun()

                if c.button("Reject", key=f"reject_{sid}"):
                    st.session_state.accepted[sid] = "Rejected"
                    try:
                        record_action("reject", {"standard_id": sid})
                    except Exception:
                        pass
                    st.rerun()

# -------------------------------------------------------------------
# Recommendations
# -------------------------------------------------------------------
elif page == "Recommendations":
    st.header("📊 Ranked Recommendations")

    if not st.session_state.analysis:
        st.info("Run an analysis first.")
    else:
        recs = st.session_state.analysis.get("recommendations", [])

        if not recs:
            st.warning("No recommendations available.")
        else:
            rows = []
            for x in recs:
                rows.append(
                    {
                        "Standard": x.get("standard_id", ""),
                        "Title": x.get("title", ""),
                        "Score": x.get("score", 0),
                        "Status": x.get("status", ""),
                        "Type": x.get("classification", ""),
                        "Sector": x.get("sector", ""),
                        "Evidence": "; ".join(
                            map(str, x.get("evidence", []))
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                "Ranking combines semantic/lexical similarity, product/property "
                "compatibility and standards metadata. The project remains "
                "offline-first and does not claim authoritative BIS compliance."
            )

# -------------------------------------------------------------------
# Compliance
# -------------------------------------------------------------------
elif page == "Compliance":
    st.header("✅ Compliance Gap Analysis")

    if not st.session_state.analysis:
        st.info("Run an analysis first.")
    else:
        result = st.session_state.analysis
        comp = result.get("compliance", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Coverage", f"{comp.get('coverage', 0)}%")
        c2.metric("Missing", len(comp.get("missing", [])))
        c3.metric("Warnings", len(comp.get("warnings", [])))

        st.subheader("Checklist")

        for item in comp.get("items", []):
            status = item.get("status", "Warning")
            icon = (
                "✅" if status == "Met"
                else "⚠️" if status == "Warning"
                else "❌"
            )
            st.write(
                f"{icon} **{item.get('requirement', 'Requirement')}** — {status}"
            )

        if comp.get("missing"):
            st.subheader("Missing / incomplete requirements")
            for item in comp["missing"]:
                st.error(item)

        if comp.get("warnings"):
            st.subheader("Warnings")
            for item in comp["warnings"]:
                st.warning(item)

        try:
            pdf = build_pdf_report(result)
            st.download_button(
                "📄 Download PDF compliance report",
                data=pdf,
                file_name="smartstandards_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error("PDF report generation failed.")
            st.exception(exc)

        try:
            csv = build_csv_report(result)
            st.download_button(
                "📊 Download Excel-compatible CSV",
                data=csv,
                file_name="compliance.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error("CSV report generation failed.")
            st.exception(exc)

# -------------------------------------------------------------------
# Specification Generator
# -------------------------------------------------------------------
elif page == "Specification Generator":
    st.header("📝 Procurement Specification Generator")

    if not st.session_state.analysis:
        st.info("Run an analysis first.")
    else:
        result = st.session_state.analysis

        st.text_area(
            "Ready-to-edit procurement specification",
            result.get("generated_specification", ""),
            height=550,
        )

        st.caption(
            "This is a decision-support draft. Verify standard numbers, "
            "editions, clauses, QCOs, certification requirements and "
            "regulatory notifications against authoritative sources "
            "before issuing a tender."
        )

# -------------------------------------------------------------------
# Standards Library
# -------------------------------------------------------------------
elif page == "Standards Library":
    st.header("📚 Searchable Standards Library")

    q = st.text_input(
        "Search standards",
        placeholder="cable, helmet, water, safety...",
    )

    status_values = sorted(
        standards["status"].dropna().astype(str).unique().tolist()
    )
    sector_values = sorted(
        standards["sector"].dropna().astype(str).unique().tolist()
    )

    status_filter = st.multiselect(
        "Status",
        status_values,
        default=status_values,
    )
    sector_filter = st.multiselect(
        "Sector",
        sector_values,
        default=sector_values,
    )

    df = standards[
        standards["status"].isin(status_filter)
        & standards["sector"].isin(sector_filter)
    ].copy()

    if q.strip():
        q_pattern = re.escape(q.strip())
        mask = df.astype(str).apply(
            lambda row: row.str.contains(
                q_pattern,
                case=False,
                na=False,
                regex=True,
            ).any(),
            axis=1,
        )
        df = df[mask]

    columns = [
        "standard_id",
        "title",
        "sector",
        "status",
        "classification",
        "scope",
    ]

    st.caption(f"{len(df)} standard(s) shown")
    st.dataframe(
        df[columns],
        use_container_width=True,
        hide_index=True,
    )

# -------------------------------------------------------------------
# Knowledge Graph
# -------------------------------------------------------------------
elif page == "Knowledge Graph":
    st.header("🕸️ Standards Knowledge Graph")

    standard_ids = standards["standard_id"].dropna().astype(str).tolist()

    if not standard_ids:
        st.warning("No standards available.")
    else:
        sid = st.selectbox("Choose a standard", standard_ids)

        try:
            graph = standard_graph(standards)
            fig = graph_figure(graph, sid)
            st.plotly_chart(fig, use_container_width=True)

            row = standards[
                standards["standard_id"].astype(str) == sid
            ].iloc[0]

            st.write(f"**{row['standard_id']}** — {row['title']}")
            st.write(row["scope"])

        except Exception as exc:
            st.error("Unable to generate the knowledge graph.")
            st.exception(exc)

# -------------------------------------------------------------------
# History
# -------------------------------------------------------------------
else:
    st.header("🕘 Reviewer / Audit History")

    try:
        history = load_history()
    except Exception as exc:
        history = []
        st.error("Unable to load history.")
        st.exception(exc)

    if history:
        st.dataframe(
            pd.DataFrame(history),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No actions recorded yet.")