import os
import re
from typing import Any, Iterable

import dotenv

import streamlit as st

from langchain_groq import ChatGroq
try:
    from langchain.agents import create_react_agent
    _HAS_CREATE_REACT_AGENT = True
except ImportError:
    _HAS_CREATE_REACT_AGENT = False

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from fpdf import FPDF



# =====================
# Config & Setup
# =====================

dotenv.load_dotenv()

st.set_page_config(
    page_title="AI Research & Report Generator",
    page_icon="📑",
    layout="wide",
)

st.title("📑 AI Research & Report Generator (Groq + LangChain)")

st.markdown(
    """
Give a topic, choose your **level**, and let the AI:
1. Research the web & Wikipedia  
2. Organize the knowledge  
3. Generate a structured report for you  
"""
)

st.markdown(
    """
    <style>
    .research-ai-hero {
        padding: 1.75rem 2.25rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(33,61,124,0.95), rgba(16,31,62,0.92));
        box-shadow: 0 20px 40px rgba(15,23,42,0.25);
        color: #f8fafc;
        border: 1px solid rgba(148,163,184,0.25);
    }
    .research-ai-hero h2 {
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
    }
    .research-ai-hero p {
        margin: 0.3rem 0 0.8rem;
        font-size: 1.02rem;
        color: rgba(226,232,240,0.88);
    }
    .research-ai-hero.secondary {
        background: rgba(15,23,42,0.82);
        border: 1px solid rgba(148,163,184,0.28);
    }
    .hero-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .hero-list li {
        padding: 0.45rem 0 0.35rem;
        border-bottom: 1px dashed rgba(148,163,184,0.35);
        font-size: 0.95rem;
        color: rgba(226,232,240,0.9);
    }
    .hero-list li:last-child {
        border-bottom: none;
    }
    .feature-card {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        border: 1px solid rgba(148,163,184,0.2);
        box-shadow: 0 6px 16px rgba(15,23,42,0.12);
        height: 100%;
    }
    .feature-card h4 {
        margin-top: 0.4rem;
        margin-bottom: 0.45rem;
        font-size: 1.05rem;
        color: #e2e8f0;
    }
    .feature-card p {
        font-size: 0.92rem;
        color: rgba(226,232,240,0.8);
        margin: 0;
    }
    .feature-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: rgba(148,163,184,0.2);
        font-size: 1.3rem;
    }
    .report-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 18px 35px rgba(15,23,42,0.08);
        border: 1px solid rgba(148,163,184,0.25);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111c36 100%);
        color: #e2e8f0;
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    button[data-baseweb="button"] {
        border-radius: 999px !important;
        padding: 0.45rem 1.4rem !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

hero_cols = st.columns([1.4, 1])
with hero_cols[0]:
    st.markdown(
        """
        <div class="research-ai-hero">
            <h2>Research-quality reports in minutes</h2>
            <p>Leverage Groq's lightning-fast LLM with trusted web tools to craft structured research summaries tailored to your audience level.</p>
            <p style="margin-top:0.6rem; color: rgba(226,232,240,0.78); font-size: 0.9rem;">Step away from scattered notes—receive polished insights with citations, adaptable depth, and export-ready layout.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_cols[1]:
    st.markdown(
        """
        <div class="research-ai-hero secondary">
            <h3 style="margin-bottom:0.6rem; font-size:1.25rem;">Workflow Snapshot</h3>
            <ul class="hero-list">
                <li>🎯 Choose learning depth (Beginner ➜ Advanced).</li>
                <li>🧭 Mix web search + Wikipedia context automatically.</li>
                <li>📝 Receive structured sections with key takeaways.</li>
                <li>📄 Export a polished PDF in one click.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

feature_data = [
    {
        "icon": "🔍",
        "title": "Evidence-backed research",
        "body": "Agents blend DuckDuckGo search with Wikipedia grounding to minimise hallucinations."
    },
    {
        "icon": "🎯",
        "title": "Audience-aware writing",
        "body": "Beginner, intermediate, or advanced—tone and depth adapt to your choice."
    },
    {
        "icon": "📤",
        "title": "Ready to share",
        "body": "Download clean PDFs, reuse markdown, or iterate further inside Streamlit."
    },
]

feature_cols = st.columns(len(feature_data))
for col, item in zip(feature_cols, feature_data):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{item['icon']}</div>
                <h4>{item['title']}</h4>
                <p>{item['body']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if not os.environ.get("GROQ_API_KEY"):
    st.warning("⚠️ GROQ_API_KEY not found. Set it in a .env file or environment variable.")

# =====================
# Helpers
# =====================

def get_level_description(level: str) -> str:
    if level == "Beginner":
        return (
            "Explain as if to a 1st–2nd year student. "
            "Use simple language, basic examples, and avoid heavy math or jargon."
        )
    elif level == "Intermediate":
        return (
            "Explain for a B.Tech AI & DS student in 3rd–4th year. "
            "Use technical terms where needed, include some depth, and practical examples."
        )
    else:  # Advanced
        return (
            "Explain for someone preparing for research or interviews. "
            "Include deeper technical details, trade-offs, and where relevant, math/architecture."
        )


REPORT_PROMPT = """
You are an expert research assistant for a B.Tech AI & DS student.

Adapt your explanation to this level:
{level_description}

Your task:
1. Use the available tools (web search, Wikipedia) to gather information.
2. Then write a clear, well-structured report for the given topic.

Report format:
- Title
- Introduction (3–5 sentences)
- Main Sections with headings and bullet points
- If it's a comparison, include a comparison table (in markdown if needed)
- Conclusion (2–4 sentences)
- References (list the main sources or article titles you used)

Write in simple, professional English according to the chosen level.
Avoid hallucinating. If something is unclear or conflicting, say so.
"""


bold_pattern = re.compile(r"\*\*(.+?)\*\*")


def _strip_markdown(text: str) -> str:
    return bold_pattern.sub(lambda m: m.group(1), text)


def _write_heading(pdf: FPDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Times", style="B", size=14)
    pdf.multi_cell(0, 8, text)
    pdf.ln(1)


def _write_paragraph(pdf: FPDF, text: str) -> None:
    pdf.set_font("Times", size=12)
    pdf.multi_cell(0, 6, _strip_markdown(text))
    pdf.ln(1)


def _write_bullet(pdf: FPDF, text: str) -> None:
    pdf.set_font("Times", size=12)
    pdf.multi_cell(0, 6, f"- {_strip_markdown(text)}")
    pdf.ln(0.5)


def _write_label_block(pdf: FPDF, label: str, body: str | None, bullet: bool) -> None:
    prefix = "- " if bullet else ""
    pdf.set_font("Times", style="B", size=12)
    pdf.multi_cell(0, 6, prefix + label)
    if body:
        pdf.set_font("Times", size=12)
        indent = "    " if bullet else ""
        pdf.multi_cell(0, 6, indent + _strip_markdown(body))
    pdf.ln(0.5)


def create_pdf(report_text: str, title: str = "AI_Report") -> bytes:
    """Render the AI report into a professionally aligned PDF without markdown artifacts."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", style="B", size=16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(4)

    for raw_line in report_text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            pdf.ln(3)
            continue

        bullet = False
        if stripped.startswith("- ") or stripped.startswith("* "):
            bullet = True
            stripped = stripped[2:].strip()

        heading_match = re.fullmatch(r"\*\*(.+)\*\*", stripped)
        if heading_match and not bullet:
            _write_heading(pdf, heading_match.group(1).strip())
            continue

        label_match = re.match(r"\*\*(.+?)\*\*(.*)$", stripped)
        if label_match:
            label = label_match.group(1).strip()
            body = label_match.group(2).lstrip(" :").strip() or None
            label_text = f"{label}:" if body else label
            _write_label_block(pdf, label_text, body, bullet)
            continue

        if bullet:
            _write_bullet(pdf, stripped)
        else:
            _write_paragraph(pdf, stripped)

    return pdf.output(dest="S").encode("latin-1")


# =====================
# LangChain Agent Setup (cached)
# =====================

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


@st.cache_resource
def get_tools():
    search = DuckDuckGoSearchRun()
    wiki_api = WikipediaAPIWrapper()
    wiki = WikipediaQueryRun(api_wrapper=wiki_api)
    return [search, wiki]


@st.cache_resource
def get_agent():
    llm = get_llm()
    tools = get_tools()
    if not _HAS_CREATE_REACT_AGENT:
        return None
    tool_descriptions = "\n".join(
        f"- {tool.name}: {getattr(tool, 'description', 'No description provided.')}"
        for tool in tools
    )
    tool_names = ", ".join(tool.name for tool in tools)

    system_instructions = f"""You are an expert research assistant for a B.Tech AI & DS student.

Always reason step by step, citing tools when used. Available tools:
{tool_descriptions}

When you take an action, follow the Thought / Action / Action Input / Observation protocol.
Only choose from: {tool_names}

At the end, produce a polished report obeying the requested format.
"""

    try:
        return create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_instructions,
        )
    except TypeError:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_instructions),
                ("human", "Question: {input}"),
                MessagesPlaceholder("messages"),
            ]
        )

        return create_react_agent(model=llm, tools=tools, prompt=prompt)


def _extract_text(messages: Iterable[Any]) -> str:
    for message in reversed(list(messages)):
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [piece.get("text", "") for piece in content if isinstance(piece, dict)]
            if parts:
                return "".join(parts)
    return ""


def generate_report(topic: str, level: str) -> str:
    level_desc = get_level_description(level)
    prompt = REPORT_PROMPT.format(level_description=level_desc)
    full_input = (
        f"{prompt}\n\n"
        f"Now generate a detailed report for this topic:\n\n"
        f"Topic: {topic}"
    )

    agent = get_agent()
    if agent is not None:
        result = agent.invoke({"input": full_input})

        if isinstance(result, str):
            return result

        output = result.get("output") if isinstance(result, dict) else None
        if isinstance(output, str) and output.strip():
            return output

        if isinstance(result, dict):
            messages = result.get("messages", [])
            fallback = _extract_text(messages)
            if fallback:
                return fallback

        return ""

    llm = get_llm()
    response = llm.invoke(full_input)
    if hasattr(response, "content"):
        content = response.content
        return content if isinstance(content, str) else str(content)
    return str(response)


# =====================
# Streamlit UI
# =====================

st.sidebar.header("⚙️ Settings")

level = st.sidebar.radio(
    "Research mode (depth):",
    ["Beginner", "Intermediate", "Advanced"],
    index=1,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 Start with *Intermediate* for balanced depth, then iterate for Beginner summaries or Advanced deep dives.",
)
st.sidebar.caption("Need citations? Reference titles appear in the report footer.")

topic = st.text_input(
    "Enter a topic for research:",
    placeholder="e.g., Transfer Learning in Deep Learning, RAG with LangChain, CNN vs Vision Transformers, etc.",
)

generate_button = st.button("🚀 Generate Report")

if "report_text" not in st.session_state:
    st.session_state.report_text = ""
    st.session_state.last_topic = ""

if generate_button:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Researching and generating your report..."):
            try:
                report = generate_report(topic.strip(), level)
                st.session_state.report_text = report
                st.session_state.last_topic = topic.strip()
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

# Display report
if st.session_state.report_text:
    st.markdown("## 📄 Generated Report")
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown(st.session_state.report_text)
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF export
    pdf_bytes = create_pdf(
        st.session_state.report_text,
        title=f"Report - {st.session_state.last_topic}",
    )

    _, download_col = st.columns([3, 1])
    with download_col:
        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_bytes,
            file_name=f"report_{st.session_state.last_topic.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

st.markdown(
    "<hr/><p style='text-align:center; font-size: 0.8rem;'>Built by MOHAMAD AADIL J · B.Tech AI & DS · Gemini + LangChain + Streamlit</p>",
    unsafe_allow_html=True,
)
