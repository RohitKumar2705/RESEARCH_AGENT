"""
Research Paper Analyzer + Chapter Summarizer + Q&A Chat
========================================================
Run:  python agent.py

3 Modes:
  1. Full Analysis       — full multi-agent research brief
  2. Chapter Summary     — pick a chapter, get instant brief
  3. Ask Questions       — chat with the paper

Install:
    pip install langgraph groq pdfplumber python-dotenv
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import TypedDict, Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────────
# CHAPTER MAP  — page ranges for each chapter
# ─────────────────────────────────────────────

CHAPTER_MAP = {
    1: {"title": "Thinking",                  "pages": (6,  13)},
    2: {"title": "Consciousness",             "pages": (13, 19)},
    3: {"title": "Learning and Growth",       "pages": (19, 22)},
    4: {"title": "Personal Life Choices",     "pages": (22, 27)},
    5: {"title": "Organizational Life Choices","pages": (27, 32)},
    6: {"title": "Choices About Society",     "pages": (32, 35)},
    7: {"title": 'The "Science" of God',      "pages": (35, 39)},
    8: {"title": 'The "Poetry" of God',       "pages": (39, 101)},
}

# ─────────────────────────────────────────────
# AUTO PDF FINDER
# ─────────────────────────────────────────────

def find_pdf(filename: str = None) -> str:
    script_dir = Path(__file__).parent.resolve()
    if filename:
        full_path = script_dir / filename
        if full_path.exists():
            return str(full_path)
        print(f" Could not find: {full_path}")
        sys.exit(1)

    pdf_files = list(script_dir.glob("*.pdf"))
    if len(pdf_files) == 0:
        print(f" No PDF found in: {script_dir}")
        sys.exit(1)
    if len(pdf_files) == 1:
        print(f"✅ Found PDF: {pdf_files[0].name}")
        return str(pdf_files[0])

    print(f"\nMultiple PDFs found:")
    for i, f in enumerate(pdf_files):
        print(f"  [{i+1}] {f.name}")
    choice = input("\nEnter number: ").strip()
    return str(pdf_files[int(choice) - 1])


# ─────────────────────────────────────────────
# PDF PROCESSOR
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> dict:
    try:
        import pdfplumber
    except ImportError:
        print(" Run:  pip install pdfplumber")
        sys.exit(1)

    pages_text = []
    metadata   = {}

    with pdfplumber.open(pdf_path) as pdf:
        if pdf.metadata:
            metadata = {
                "title":  pdf.metadata.get("Title",  "Unknown"),
                "author": pdf.metadata.get("Author", "Unknown"),
            }
        for page in pdf.pages:
            text = page.extract_text()
            pages_text.append(
                re.sub(r'\n{3,}', '\n\n', text).strip() if text else ""
            )

    full_text = "\n\n".join([t for t in pages_text if t])
    print(f" Extracted {len(pages_text)} pages | {len(full_text):,} chars")
    return {
        "full_text":  full_text,
        "pages":      pages_text,       # list indexed by page number
        "total_pages": len(pages_text),
        "metadata":   metadata,
    }


def get_chapter_text(pages: list, chapter_num: int) -> str:
    """Extract text for a specific chapter using CHAPTER_MAP."""
    if chapter_num not in CHAPTER_MAP:
        return ""
    start, end = CHAPTER_MAP[chapter_num]["pages"]
    end = min(end, len(pages))
    text = "\n\n".join([pages[i] for i in range(start, end) if pages[i]])
    return text


def get_agent_text(full_text: str, max_chars: int = 15000) -> str:
    if len(full_text) <= max_chars:
        return full_text
    first = int(max_chars * 0.6)
    last  = max_chars - first
    return full_text[:first] + "\n\n[...trimmed...]\n\n" + full_text[-last:]


def pre_extract_citations(full_text: str) -> list:
    citations = []
    numbered  = re.findall(r'\[\d+\]\s+.{10,120}', full_text)
    citations.extend(numbered[:50])
    ref_match = re.search(
        r'(?:References|Bibliography|Works Cited)\s*\n([\s\S]{100,3000})',
        full_text, re.IGNORECASE)
    if ref_match:
        lines = [l.strip() for l in ref_match.group(1).split('\n') if len(l.strip()) > 20]
        citations.extend(lines[:40])
    seen, unique = set(), []
    for c in citations:
        if c not in seen:
            seen.add(c); unique.append(c)
    return unique


def load_pdf(pdf_path: str) -> dict:
    print(f"\n📄 Loading PDF: {pdf_path}")
    extracted  = extract_text_from_pdf(pdf_path)
    agent_text = get_agent_text(extracted["full_text"])
    pre_cites  = pre_extract_citations(extracted["full_text"])
    print(f"   Agent text  : {len(agent_text):,} chars")
    print(f"   Citations   : {len(pre_cites)} pre-found\n")
    return {
        "full_text":     extracted["full_text"],
        "pages":         extracted["pages"],
        "agent_text":    agent_text,
        "pre_citations": pre_cites,
        "metadata":      extracted["metadata"],
    }


# ─────────────────────────────────────────────
# GROQ API
# ─────────────────────────────────────────────

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(" GROQ_API_KEY not found.")
            print("   Add to .env file:  GROQ_API_KEY=your_key")
            print("   Get free key at:   https://console.groq.com")
            sys.exit(1)
        _client = Groq(api_key=api_key)
    return _client

MODEL = "llama-3.3-70b-versatile"


def call_groq(system: str, user: str) -> str:
    for attempt in range(3):
        try:
            r = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            return r.choices[0].message.content
        except Exception as e:
            print(f"     API error (attempt {attempt+1}/3): {e}")
            if attempt == 2:
                return "{}"
    return "{}"


def call_groq_chat(messages: list) -> str:
    for attempt in range(3):
        try:
            r = get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=1500,
                temperature=0.5,
            )
            return r.choices[0].message.content
        except Exception as e:
            print(f"     API error (attempt {attempt+1}/3): {e}")
            if attempt == 2:
                return "Sorry, could not get a response."
    return ""


def safe_json(raw: str, fallback):
    try:
        clean = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        clean = re.sub(r'\s*```$', '', clean).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return fallback


# ─────────────────────────────────────────────
# MODE 2 — CHAPTER SUMMARIZER  ← NEW
# ─────────────────────────────────────────────

def chapter_summary_mode(pdf_data: dict):
    """
    Pick any chapter and get an instant structured brief.
    Shows: core message, key ideas, practical takeaway.
    """
    pages = pdf_data["pages"]

    while True:
        print("\n" + "="*54)
        print("  CHAPTER SUMMARY MODE")
        print("="*54)
        print("  Available chapters:")
        for num, info in CHAPTER_MAP.items():
            print(f"  [{num}] Chapter {num} — {info['title']}")
        print("  [0] Go back to main menu")
        print("="*54)

        choice = input("\nEnter chapter number: ").strip()

        if choice == "0":
            break

        try:
            chapter_num = int(choice)
            if chapter_num not in CHAPTER_MAP:
                print(" Invalid chapter number. Try again.")
                continue
        except ValueError:
            print(" Please enter a number.")
            continue

        chapter_title = CHAPTER_MAP[chapter_num]["title"]
        chapter_text  = get_chapter_text(pages, chapter_num)

        if not chapter_text.strip():
            print(f" Could not extract text for Chapter {chapter_num}.")
            continue

        print(f"\n Summarizing Chapter {chapter_num}: {chapter_title}...")
        print("   Please wait...\n")

        # Call Groq to generate structured chapter brief
        raw = call_groq(
            system="""You are an expert book summarizer.
Your job is to give a very short, clear, and useful chapter brief.

Respond in this EXACT format (use these exact headings):

CORE MESSAGE:
[One sentence — the single most important idea of this chapter]

KEY IDEAS:
1. [idea 1 — max 2 lines]
2. [idea 2 — max 2 lines]
3. [idea 3 — max 2 lines]
4. [idea 4 — max 2 lines]
5. [idea 5 — max 2 lines]

PRACTICAL TAKEAWAY:
[One paragraph — how to apply this chapter in real life, max 3 sentences]

BEST QUOTE OR EXAMPLE FROM CHAPTER:
[One powerful line or example from the chapter]

Keep everything short, simple, and easy to understand.""",

            user=f"""Summarize this chapter briefly and clearly.

Chapter: {chapter_num} — {chapter_title}

Chapter Content:
{chapter_text}"""
        )

        # Display the result nicely
        print("\n" + "╔" + "═"*52 + "╗")
        print(f"║   CHAPTER {chapter_num}: {chapter_title.upper():<38}║")
        print("╚" + "═"*52 + "╝")
        print(raw)
        print("\n" + "─"*54)

        # Save to file
        output_path = Path(__file__).parent / f"chapter_{chapter_num}_summary.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"CHAPTER {chapter_num}: {chapter_title}\n")
            f.write("="*54 + "\n\n")
            f.write(raw)
        print(f" Saved to: chapter_{chapter_num}_summary.txt")

        again = input("\n📖 Summarize another chapter? (y/n): ").strip().lower()
        if again != "y":
            break


# ─────────────────────────────────────────────
# MODE 3 — Q&A CHAT
# ─────────────────────────────────────────────

def qa_chat_mode(pdf_data: dict):
    full_text = pdf_data["full_text"]
    metadata  = pdf_data["metadata"]
    paper_context = get_agent_text(full_text, max_chars=20000)

    print("\n" + "="*54)
    print("  Q&A CHAT MODE")
    print(f"    Paper: {metadata.get('title', 'Unknown')}")
    print("="*54)
    print("  Ask anything about the paper.")
    print("  Commands: 'summarize' | 'clear' | 'quit'")
    print("  Tip: Ask about specific chapters too!")
    print("  e.g. 'What is Chapter 3 about?'")
    print("="*54 + "\n")

    system_prompt = f"""You are an expert assistant for the book "The Road Less Traveled" by M. Scott Peck.
You have read the entire book and can answer any question about it.

BOOK CONTENT:
{paper_context}

CHAPTER LIST:
Chapter 1: Thinking
Chapter 2: Consciousness
Chapter 3: Learning and Growth
Chapter 4: Personal Life Choices
Chapter 5: Organizational Life Choices
Chapter 6: Choices About Society
Chapter 7: The "Science" of God
Chapter 8: The "Poetry" of God

Instructions:
- Answer clearly and specifically based on the book content.
- If not in the book, say "This is not covered in the book."
- Use bullet points for lists.
- Keep answers concise but helpful.
- When asked about a chapter, give a structured response with key ideas.
"""

    chat_history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\n You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        if user_input.lower() == "clear":
            chat_history = [{"role": "system", "content": system_prompt}]
            print(" Chat history cleared.")
            continue

        if user_input.lower() == "summarize":
            user_input = "Give me a concise summary of this entire book in 150-200 words."

        chat_history.append({"role": "user", "content": user_input})
        print("\n Agent: ", end="", flush=True)

        response = call_groq_chat(chat_history)
        print(response)

        chat_history.append({"role": "assistant", "content": response})

        # Trim history if too long
        if len(chat_history) > 22:
            chat_history = [chat_history[0]] + chat_history[-20:]


# ─────────────────────────────────────────────
# MODE 1 — FULL ANALYSIS (LangGraph)
# ─────────────────────────────────────────────

from langgraph.graph import StateGraph, END

class ResearchState(TypedDict):
    paper_text:    str
    pre_citations: list
    metadata:      dict
    analysis:      Optional[dict]
    summary:       Optional[str]
    citations:     Optional[list]
    insights:      Optional[dict]
    review_scores: dict
    retry_counts:  dict
    final_brief:   Optional[str]


def boss_node(state):
    print(" [Boss Agent] Starting pipeline...")
    return {**state, "review_scores": {}, "retry_counts": {},
            "analysis": None, "summary": None,
            "citations": None, "insights": None, "final_brief": None}

def analyzer_node(state):
    retry = state["retry_counts"].get("analysis", 0)
    print(f"🔬 [Paper Analyzer] Running... (attempt {retry+1})")
    raw = call_groq(
        system="""Extract structured info. Respond ONLY with valid JSON:
{"problem_statement":"...","methodology":"...","key_findings":["...","...","..."],"experiments":"...","conclusion":"..."}""",
        user=f"Analyze:\n\n{state['paper_text']}"
    )
    return {**state, "analysis": safe_json(raw, {"raw_output": raw})}

def summary_node(state):
    retry = state["retry_counts"].get("summary", 0)
    print(f" [Summary Agent] Running... (attempt {retry+1})")
    raw = call_groq(
        system="Write a 150-200 word executive summary. Return ONLY the summary text.",
        user=f"Analysis:\n{json.dumps(state.get('analysis',{}))}\n\nPaper:\n{state['paper_text'][:4000]}"
    )
    return {**state, "summary": raw.strip()}

def citation_node(state):
    retry = state["retry_counts"].get("citations", 0)
    print(f" [Citation Extractor] Running... (attempt {retry+1})")
    pre = "\n".join(state.get("pre_citations", [])[:20]) or "None"
    raw = call_groq(
        system='Respond ONLY with a JSON array of citation strings. No markdown.',
        user=f"Hints:\n{pre}\n\nPaper:\n{state['paper_text'][-5000:]}"
    )
    citations = safe_json(raw, [])
    if not isinstance(citations, list):
        citations = [l.strip() for l in raw.split("\n") if len(l.strip()) > 10]
    return {**state, "citations": citations}

def insights_node(state):
    retry = state["retry_counts"].get("insights", 0)
    print(f"💡 [Insights Agent] Running... (attempt {retry+1})")
    raw = call_groq(
        system="""Respond ONLY with valid JSON:
{"practical_takeaways":["...","...","..."],"field_implications":"...","potential_applications":["...","..."],"limitations":"..."}""",
        user=f"Analysis:\n{json.dumps(state.get('analysis',{}))}\n\nPaper:\n{state['paper_text'][:3000]}"
    )
    return {**state, "insights": safe_json(raw, {"raw_output": raw})}

def make_review_node(target: str):
    def review(state):
        content_map = {
            "analysis":  json.dumps(state.get("analysis",  {})),
            "summary":   state.get("summary",  ""),
            "citations": json.dumps(state.get("citations", [])),
            "insights":  json.dumps(state.get("insights",  {})),
        }
        print(f" [Review Agent] Reviewing {target}...")
        raw    = call_groq(
            system='Score 1-10. Respond ONLY with JSON: {"score":8,"feedback":"reason"}',
            user=f"Review this {target}:\n\n{content_map[target][:2000]}"
        )
        result  = safe_json(raw, {"score": 7, "feedback": "defaulting to pass"})
        score   = int(result.get("score", 7))
        print(f"   Score: {score}/10 — {result.get('feedback','')}")
        new_scores  = {**state["review_scores"],  target: score}
        new_retries = {**state["retry_counts"]}
        if score < 7:
            new_retries[target] = new_retries.get(target, 0) + 1
        return {**state, "review_scores": new_scores, "retry_counts": new_retries}
    review.__name__ = f"review_{target}_node"
    return review

def combine_node(state):
    print("📋 [Boss Agent] Combining outputs...")
    meta = state.get("metadata", {})
    a    = state.get("analysis", {})
    s    = state.get("review_scores", {})
    brief = f"""
╔══════════════════════════════════════════════════════╗
║              RESEARCH BRIEF                         ║
╚══════════════════════════════════════════════════════╝
Title  : {meta.get('title','Unknown')}
Author : {meta.get('author','Unknown')}

ANALYSIS (score: {s.get('analysis','?')}/10)
Problem : {a.get('problem_statement','N/A')}
Method  : {a.get('methodology','N/A')}
Findings:
{chr(10).join(f"  • {f}" for f in a.get('key_findings',[]))}
Conclusion: {a.get('conclusion','N/A')}

SUMMARY (score: {s.get('summary','?')}/10)
{state.get('summary','')}

CITATIONS (score: {s.get('citations','?')}/10)
{chr(10).join(f"  [{i+1}] {c}" for i,c in enumerate(state.get('citations',[])[:20]))}

INSIGHTS (score: {s.get('insights','?')}/10)
Takeaways:
{chr(10).join(f"  ✓ {t}" for t in state.get('insights',{}).get('practical_takeaways',[]))}
Implications: {state.get('insights',{}).get('field_implications','N/A')}
Applications:
{chr(10).join(f"  → {a}" for a in state.get('insights',{}).get('potential_applications',[]))}
Limitations: {state.get('insights',{}).get('limitations','N/A')}

SCORES: {s}
""".strip()
    print(" Brief ready!")
    return {**state, "final_brief": brief}

def make_router(target, next_node, retry_node):
    def route(state):
        if state["review_scores"].get(target, 0) >= 7 or state["retry_counts"].get(target, 0) >= 2:
            return next_node
        return retry_node
    route.__name__ = f"route_{target}"
    return route

def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("boss",             boss_node)
    g.add_node("analyzer",         analyzer_node)
    g.add_node("review_analysis",  make_review_node("analysis"))
    g.add_node("summary",          summary_node)
    g.add_node("review_summary",   make_review_node("summary"))
    g.add_node("citations",        citation_node)
    g.add_node("review_citations", make_review_node("citations"))
    g.add_node("insights",         insights_node)
    g.add_node("review_insights",  make_review_node("insights"))
    g.add_node("combine",          combine_node)
    g.set_entry_point("boss")
    g.add_edge("boss", "analyzer")
    g.add_edge("analyzer", "review_analysis")
    g.add_conditional_edges("review_analysis",
        make_router("analysis","summary","analyzer"),
        {"summary":"summary","analyzer":"analyzer"})
    g.add_edge("summary","review_summary")
    g.add_conditional_edges("review_summary",
        make_router("summary","citations","summary"),
        {"citations":"citations","summary":"summary"})
    g.add_edge("citations","review_citations")
    g.add_conditional_edges("review_citations",
        make_router("citations","insights","citations"),
        {"insights":"insights","citations":"citations"})
    g.add_edge("insights","review_insights")
    g.add_conditional_edges("review_insights",
        make_router("insights","combine","insights"),
        {"combine":"combine","insights":"insights"})
    g.add_edge("combine", END)
    return g.compile()

def run_full_analysis(pdf_data: dict):
    app = build_graph()
    initial_state = ResearchState(
        paper_text=pdf_data["agent_text"], pre_citations=pdf_data["pre_citations"],
        metadata=pdf_data["metadata"], analysis=None, summary=None,
        citations=None, insights=None, review_scores={}, retry_counts={}, final_brief=None,
    )
    print(" Starting multi-agent pipeline...\n")
    final_state = app.invoke(initial_state)
    brief = final_state.get("final_brief", "No output generated.")
    output_path = Path(__file__).parent / "research_brief.txt"
    output_path.write_text(brief, encoding="utf-8")
    print(f"\n Saved to: {output_path}")
    print("\n" + "="*54)
    print(brief[:600])
    print("...")


# ─────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pdf_path = find_pdf()
    pdf_data = load_pdf(pdf_path)

    while True:
        print("\n" + "╔" + "═"*52 + "╗")
        print("║       RESEARCH PAPER ANALYZER                   ║")
        print("║       The Road Less Traveled                    ║")
        print("╠" + "═"*52 + "╣")
        print("║  [1]  Full Analysis    — complete research brief║")
        print("║  [2]  Chapter Summary  — pick a chapter, get    ║")
        print("║                          instant brief          ║")
        print("║  [3]  Ask Questions    — chat with the book     ║")
        print("║  [0]  Exit                                      ║")
        print("╚" + "═"*52 + "╝")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            run_full_analysis(pdf_data)

        elif choice == "2":
            chapter_summary_mode(pdf_data)

        elif choice == "3":
            qa_chat_mode(pdf_data)

        elif choice == "0":
            print("\n👋 Goodbye!")
            break

        else:
            print(" Invalid choice. Enter 1, 2, 3, or 0.")