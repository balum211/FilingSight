"""FilingSight: chat with the Uber and Lyft FY2021 10-K filings.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from filingsight import config, llm  # noqa: E402
from filingsight.embeddings import EMBEDDING_MODEL_NAME  # noqa: E402
from filingsight.graph import ask  # noqa: E402
from filingsight.icons import icon  # noqa: E402
from filingsight.index import load_index  # noqa: E402

st.set_page_config(page_title="FilingSight · 10-K answers with citations", page_icon="📑", layout="wide")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eval" / "results"
SAMPLES = [
    ("Metric", "What was Uber's total revenue in 2021?"),
    ("Metric", "How many Active Riders did Lyft have in Q4 2021?"),
    ("Compare", "Compare Uber's and Lyft's revenue growth in 2021."),
    ("Strategy", "What did Lyft sell to Woven Planet?"),
    ("Risk", "How do Uber and Lyft describe the impact of Proposition 22?"),
    ("Out of scope", "What was Uber's revenue in 2023?"),
]

# ---------------------------------------------------------------- theme
# Visual system follows elayadesign/ai-design-skills (landing-page-design, Part B):
# one typeface (Manrope, Geist Mono for numbers), flat dark surfaces from the
# approved palette, Tailwind type and spacing steps, nested radii, Phosphor icons,
# cubic-bezier(0.22,1,0.36,1) motion and reduced-motion support.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');
:root{
  --bg:#181818; --surface:#1F1F1F; --card:#272727; --line:#313131; --deep:#131209;
  --text:#EDEDED; --muted:#9B9B9B; --faint:#6B6B6B;
  --accent:#7C9DFF; --accent-soft:rgba(124,157,255,.14);
  --ok:#5BD69A; --ok-soft:rgba(91,214,154,.14);
  --warn:#F2C35B; --warn-soft:rgba(242,195,91,.14);
  --bad:#F07A7A; --bad-soft:rgba(240,122,122,.14);
  --uber:#EDEDED; --lyft:#FF6FD8;
  --ease:cubic-bezier(0.22,1,0.36,1);
}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp div[data-testid="stMarkdownContainer"],
.stApp input, .stApp textarea, .stApp button{font-family:'Manrope',system-ui,sans-serif !important;}
code, pre, .mono{font-family:'Geist Mono',ui-monospace,monospace !important;}
em, i{font-style:normal;}
.stApp{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:32px; padding-bottom:96px; max-width:1200px;}
h1,h2,h3,h4{text-wrap:balance; font-weight:600 !important; letter-spacing:-0.01em;}
p, li{text-wrap:pretty;}
.ph{display:inline-block; vertical-align:-3px; flex:none;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--surface); border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container{padding-top:24px;}
.brand{display:flex; gap:12px; align-items:center; margin-bottom:4px;}
.brand-mark{width:40px; height:40px; border-radius:12px; background:var(--accent-soft); color:var(--accent);
  display:flex; align-items:center; justify-content:center; border:1px solid rgba(124,157,255,.28);}
.brand-name{font-size:20px; line-height:28px; font-weight:600; color:var(--text);}
.brand-sub{font-size:12px; line-height:16px; color:var(--muted); font-weight:500;}
.side-label{font-size:12px; line-height:16px; font-weight:600; color:var(--muted); text-transform:uppercase;
  letter-spacing:.06em; margin:24px 0 8px;}

/* status card */
.status{border:1px solid var(--line); background:var(--card); border-radius:16px; padding:12px; display:flex; flex-direction:column; gap:8px; margin-bottom:12px;}
.status-head{display:flex; align-items:center; gap:8px; font-size:14px; line-height:20px; font-weight:600; color:var(--text);}
.dot{width:8px; height:8px; border-radius:999px; flex:none;}
.dot.ok{background:var(--ok); box-shadow:0 0 0 4px var(--ok-soft);}
.dot.warn{background:var(--warn); box-shadow:0 0 0 4px var(--warn-soft);}
.dot.bad{background:var(--bad); box-shadow:0 0 0 4px var(--bad-soft);}
.status-model{font-size:12px; line-height:16px; color:var(--muted); font-family:'Geist Mono',monospace; word-break:break-all;}
.status-msg{font-size:12px; line-height:16px; color:var(--text); background:var(--surface); border:1px solid var(--line);
  border-radius:4px; padding:8px;}

/* hero */
.hero{padding:16px 0 32px;}
.eyebrow{display:inline-flex; align-items:center; gap:8px; padding:4px 12px; border-radius:999px; border:1px solid var(--line);
  background:var(--surface); color:var(--muted); font-size:12px; line-height:16px; font-weight:500; margin-bottom:16px;}
.hero h1{font-size:48px; line-height:48px; margin:0 0 16px; max-width:680px; padding:0;
  background:linear-gradient(90deg,#FFFFFF,#9B9B9B); -webkit-background-clip:text; background-clip:text; color:transparent;}
.hero p{font-size:18px; line-height:28px; color:var(--muted); max-width:680px; margin:0;}

/* kpi cards */
.kpis{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:0 0 32px;}
.kpi{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:16px; display:flex; flex-direction:column; gap:8px;
  transition:transform .5s var(--ease), border-color .3s var(--ease);}
.kpi:hover{transform:translateY(-2px); border-color:#3d3d3d;}
.kpi-top{display:flex; align-items:center; justify-content:space-between; color:var(--muted); font-size:12px; line-height:16px; font-weight:500;}
.kpi-icon{width:32px; height:32px; border-radius:8px; background:var(--card); color:var(--accent); display:flex; align-items:center; justify-content:center;}
.kpi-value{font-size:30px; line-height:36px; font-weight:600; color:var(--text); font-family:'Geist Mono',monospace !important;}
.kpi-note{font-size:12px; line-height:16px; color:var(--faint);}
@media (max-width:900px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr));} .hero h1{font-size:36px; line-height:40px;}}

/* tabs */
[data-baseweb="tab-list"]{gap:4px; background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:4px; width:fit-content;}
[data-baseweb="tab"]{height:36px; padding:0 16px; border-radius:8px; color:var(--muted); font-weight:600; font-size:14px;
  transition:background-color .3s var(--ease), color .3s var(--ease);}
[data-baseweb="tab"]:hover{color:var(--text);}
[data-baseweb="tab"][aria-selected="true"]{background:var(--card); color:var(--text);}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{display:none;}
[data-testid="stTabs"] [role="tabpanel"]{padding-top:24px;}

/* buttons */
.stButton > button{border-radius:12px; border:1px solid var(--line); background:var(--surface); color:var(--text);
  padding:8px 12px; font-weight:600; font-size:14px; line-height:20px;
  transition:transform .5s var(--ease), background-color .3s var(--ease), border-color .3s var(--ease);}
.stButton > button:hover{background:var(--card); border-color:var(--accent); color:var(--text); transform:translateY(-1px);}
.stButton > button:active{transform:scale(0.98);}
.stButton > button:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
[class*="st-key-sample"] .stButton > button{min-height:88px; text-align:left; justify-content:flex-start; align-items:flex-start;
  padding:16px; font-weight:500; font-size:14px; line-height:20px; border-radius:16px; white-space:normal;}
[class*="st-key-sample"] .stButton > button > div{justify-content:flex-start; width:100%;}
[class*="st-key-sample"] .stButton > button p{text-align:left; white-space:normal; overflow:visible; text-overflow:clip;}
.sample-tag{display:inline-flex; align-items:center; gap:4px; font-size:12px; line-height:16px; font-weight:600; color:var(--accent);
  margin:0 0 -4px 4px;}

/* chat */
[data-testid="stChatMessage"]{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:16px;
  animation:rise .6s var(--ease) both;}
[data-testid="stChatInput"]{border-radius:16px;}
[data-testid="stChatInput"] textarea{font-size:16px;}
@keyframes rise{from{opacity:0; transform:translateY(32px);} to{opacity:1; transform:none;}}

.meta{display:flex; flex-wrap:wrap; gap:8px; margin:12px 0;}
.chip{display:inline-flex; align-items:center; gap:4px; padding:2px 8px; border-radius:999px; border:1px solid var(--line);
  background:var(--card); color:var(--muted); font-size:12px; line-height:16px; font-weight:500;}
.chip.llm{color:var(--accent); background:var(--accent-soft); border-color:rgba(124,157,255,.3);}
.chip.ext{color:var(--warn); background:var(--warn-soft); border-color:rgba(242,195,91,.3);}
.chip.ref{color:var(--bad); background:var(--bad-soft); border-color:rgba(240,122,122,.3);}
.notice{display:flex; gap:12px; align-items:flex-start; padding:12px; border-radius:12px; margin-bottom:12px;
  font-size:14px; line-height:20px; border:1px solid var(--line);}
.notice.warn{background:var(--warn-soft); border-color:rgba(242,195,91,.3); color:#F7DCA0;}
.notice.bad{background:var(--bad-soft); border-color:rgba(240,122,122,.3); color:#F6B4B4;}
.notice b{color:var(--text); font-weight:600;}

/* sources */
[data-testid="stExpander"] details{border:1px solid var(--line); border-radius:12px; background:var(--bg);}
[data-testid="stExpander"] summary{font-weight:600; font-size:14px;}
.src{border:1px solid var(--line); background:var(--surface); border-radius:8px; padding:12px; margin-bottom:8px;}
.src.cited{border-color:rgba(124,157,255,.45);}
.src-head{display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:8px; font-size:14px; line-height:20px; font-weight:600; color:var(--text);}
.src-n{font-family:'Geist Mono',monospace; font-size:12px; color:var(--accent); background:var(--accent-soft); border-radius:4px; padding:0 4px;}
.co{font-size:12px; line-height:16px; font-weight:600; padding:0 8px; border-radius:999px; border:1px solid var(--line);}
.co.Uber{color:var(--uber);} .co.Lyft{color:var(--lyft); border-color:rgba(255,111,216,.35);}
.src-item{font-size:12px; line-height:16px; color:var(--muted); font-weight:500;}
.src-text{font-size:14px; line-height:20px; color:#C9C9C9; white-space:pre-wrap; max-height:160px; overflow:auto;}
.bar{height:4px; border-radius:999px; background:var(--card); width:96px; overflow:hidden; display:inline-block; vertical-align:middle;}
.bar > span{display:block; height:100%; background:var(--accent); border-radius:999px;}

/* empty state */
.empty{border:1px solid var(--line); background:var(--surface); border-radius:16px; padding:24px; margin-bottom:16px; display:flex; gap:16px; align-items:center;}
.empty-icon{width:48px; height:48px; border-radius:12px; background:var(--accent-soft); color:var(--accent); display:flex; align-items:center; justify-content:center; flex:none;}
.empty h3{font-size:18px; line-height:28px; margin:0; padding:0; font-weight:600;}
.empty p{font-size:14px; line-height:20px; margin:0; color:var(--muted);}

/* section + tables */
.section-title{display:flex; align-items:center; gap:8px; font-size:20px; line-height:28px; font-weight:600; color:var(--text); margin:0 0 4px;}
.section-sub{font-size:14px; line-height:20px; color:var(--muted); margin:0 0 16px;}
.tbl{width:100%; border-collapse:separate; border-spacing:0; border:1px solid var(--line); border-radius:16px; overflow:hidden; margin-bottom:32px; font-size:14px;}
.tbl th{background:var(--surface); color:var(--muted); font-weight:600; font-size:12px; line-height:16px; text-transform:uppercase; letter-spacing:.04em; text-align:left; padding:12px; border:none;}
.tbl td{padding:12px; border:none; border-top:1px solid var(--line); color:var(--text);}
.tbl td.num{font-family:'Geist Mono',monospace; text-align:right;}
.tbl th.num{text-align:right;}
.tbl tr.best td{background:var(--accent-soft);}
.stMarkdown table:not(.tbl){border-collapse:collapse; font-size:13px;}
.stMarkdown table:not(.tbl) th{background:var(--surface);}

/* pipeline */
.steps{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:32px;}
.step{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:16px; display:flex; flex-direction:column; gap:8px;}
.step-top{display:flex; align-items:center; gap:12px;}
.step-icon{width:40px; height:40px; border-radius:12px; background:var(--card); color:var(--accent); display:flex; align-items:center; justify-content:center;}
.step-n{font-family:'Geist Mono',monospace; font-size:12px; color:var(--faint);}
.step h4{font-size:16px; line-height:24px; margin:0; padding:0; font-weight:600;}
.step p{font-size:14px; line-height:20px; color:var(--muted); margin:0;}
.step .exit{font-size:12px; line-height:16px; color:var(--bad); display:flex; gap:4px; align-items:center;}
@media (max-width:900px){.steps{grid-template-columns:1fr;}}


/* floating nav */
.topbar{position:sticky; top:12px; z-index:50; margin:0 auto 32px; width:fit-content; max-width:100%;
  display:flex; align-items:center; gap:16px; padding:8px 8px 8px 16px; border-radius:999px;
  background:rgba(31,31,31,.72); border:1px solid var(--line); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
  animation:rise .6s var(--ease) both;}
.topbar .tb-brand{display:flex; align-items:center; gap:8px; font-size:14px; font-weight:600; color:var(--text);}
.topbar .tb-brand .ph{color:var(--accent);}
.tb-pill{white-space:nowrap; display:inline-flex; align-items:center; gap:8px; font-size:12px; line-height:16px; font-weight:500; color:var(--muted);
  padding:4px 12px; border-radius:999px; background:var(--card); border:1px solid var(--line);}
.tb-link{display:inline-flex; align-items:center; gap:4px; font-size:14px; line-height:20px; font-weight:600; color:var(--text) !important;
  text-decoration:none !important; padding:8px 12px; border-radius:999px; background:var(--card); border:1px solid var(--line);
  transition:background-color .3s var(--ease), border-color .3s var(--ease), transform .5s var(--ease);}
.tb-link:hover{border-color:var(--accent); transform:translateY(-1px);}
.tb-link:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}

/* hero grid + preview card */
.hero-grid{display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:32px; align-items:center; padding-bottom:32px;}
.hero-grid .hero{padding:0;}
.preview{background:var(--surface); border:1px solid var(--line); border-radius:24px; padding:16px; display:flex; flex-direction:column; gap:12px;
  animation:rise .8s var(--ease) .1s both;}
.pv-q{align-self:flex-end; background:var(--card); border:1px solid var(--line); border-radius:16px; padding:8px 12px; font-size:14px; line-height:20px; color:var(--text); max-width:85%;}
.pv-a{background:var(--bg); border:1px solid var(--line); border-radius:16px; padding:12px; font-size:14px; line-height:20px; color:#D6D6D6;}
.pv-a b{color:var(--text); font-weight:600;}
.pv-cite{font-family:'Geist Mono',monospace; font-size:12px; color:var(--accent); background:var(--accent-soft); border-radius:4px; padding:0 4px;}
.pv-src{display:flex; align-items:center; gap:8px; border:1px solid rgba(124,157,255,.35); background:var(--surface); border-radius:8px; padding:8px; font-size:12px; line-height:16px; color:var(--muted);}
.pv-src b{color:var(--text); font-weight:600;}
.hero-grid .hero h1{font-size:36px; line-height:40px;}
.hero h1 span{display:block; white-space:nowrap;}
.hero h1{padding:0 !important; margin:0 0 16px !important;} .hero h1 [data-testid="stHeaderActionElements"], .hero h1 a{display:none !important;}
@media (max-width:900px){.hero-grid{grid-template-columns:1fr;}}
@media (max-width:600px){.hero-grid .hero h1{font-size:30px; line-height:36px;} .tagline p{font-size:24px; line-height:32px;} .tb-brand span{display:none;} .co-stats{gap:12px;}}

/* corpus cards */
.corpus{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:0 0 32px;}
.co-card{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:16px; display:flex; gap:16px; align-items:center;
  transition:transform .5s var(--ease), border-color .3s var(--ease);}
.co-card:hover{transform:translateY(-2px); border-color:#3d3d3d;}
.co-logo{width:48px; height:48px; border-radius:12px; display:flex; align-items:center; justify-content:center; flex:none; background:var(--card);}
.co-card.uber .co-logo{color:var(--uber);} .co-card.lyft .co-logo{color:var(--lyft); background:rgba(255,111,216,.12);}
.co-name{font-size:18px; line-height:28px; font-weight:600; color:var(--text);}
.co-meta{font-size:12px; line-height:16px; color:var(--muted);}
.co-stats{margin-left:auto; display:flex; gap:24px; text-align:right;}
.co-stat .v{font-family:'Geist Mono',monospace; font-size:20px; line-height:28px; color:var(--text);}
.co-stat .l{font-size:12px; line-height:16px; color:var(--faint);}
@media (max-width:900px){.corpus{grid-template-columns:1fr;}}

/* tagline reveal */
.tagline{padding:96px 0 64px; display:flex; justify-content:center;}
.tagline p{max-width:800px; margin:0; font-size:36px; line-height:40px; font-weight:600; text-align:left; text-wrap:balance; color:var(--text);}
.tagline .w{opacity:.3; display:inline-block; animation:wordin .8s var(--ease) both; animation-delay:calc(var(--i) * 60ms + .4s);}
@supports (animation-timeline: view()){
  .tagline .w{animation-delay:0s; animation-timeline:view();
    animation-range:entry calc(var(--i) * 2.5%) cover calc(25% + var(--i) * 2.5%);}
}
@keyframes wordin{from{opacity:.3;} to{opacity:1;}}

/* skeleton */
.skel{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:16px; display:flex; flex-direction:column; gap:12px; margin-bottom:16px;}
.skel .bar-s{height:12px; border-radius:999px; background:var(--card); position:relative; overflow:hidden;}
.skel .bar-s::after{content:""; position:absolute; inset:0; transform:translateX(-100%);
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.06),transparent); animation:shimmer 1.4s var(--ease) infinite;}
.skel-head{display:flex; align-items:center; gap:8px; font-size:12px; line-height:16px; color:var(--muted); font-weight:500;}
@keyframes shimmer{to{transform:translateX(100%);}}

/* footer */
.foot{margin-top:32px; padding:24px 0; border-top:1px solid var(--line); display:flex; flex-wrap:wrap; gap:16px; align-items:center;
  justify-content:space-between; font-size:12px; line-height:16px; color:var(--faint);}
.foot a{color:var(--muted) !important; text-decoration:none !important; display:inline-flex; align-items:center; gap:4px;}
.foot a:hover{color:var(--text) !important;}
.foot .links{display:flex; gap:16px;}
[data-testid="stVegaLiteChart"]{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:16px; margin-bottom:24px;}

@media (prefers-reduced-motion: reduce){
  *{animation:none !important; transition:opacity .2s linear !important;}
  .kpi:hover, .co-card:hover, .tb-link:hover, .stButton > button:hover{transform:none;}
  .tagline .w{opacity:1 !important; animation:none !important;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def h(text) -> str:
    return html.escape(str(text))


def md_safe(text: str) -> str:
    """Escape $ so Streamlit does not render dollar amounts as LaTeX."""
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Loading the search index (first run only)...")
def warm(strategy: str):
    return len(load_index(strategy).docs)


@st.cache_data
def eval_summary():
    retr, ans = [], {}
    try:
        retr = [r["summary"] for r in json.loads((RESULTS / "retrieval.json").read_text(encoding="utf-8"))]
    except Exception:  # noqa: BLE001 - results are optional
        pass
    try:
        ans = json.loads((RESULTS / "answers.json").read_text(encoding="utf-8"))["summary"]
    except Exception:  # noqa: BLE001
        pass
    return retr, ans


def reranker_options() -> list[str]:
    opts = ["local", "none"]
    try:
        import sentence_transformers  # noqa: F401

        opts.insert(1, "cross-encoder")
    except ImportError:
        pass
    if config.llm_available():
        opts.append("llm")
    return opts


def llm_status(force: bool = False) -> llm.LLMStatus:
    """Check the Llama connection once per session (or on demand) and remember the result."""
    key = (config.llm_provider(), config.LLM_MODEL, config.llm_api_key()[-6:])
    cached = st.session_state.get("llm_status")
    if force or not cached or cached[0] != key:
        if config.llm_available():
            with st.spinner("Checking the Llama connection..."):
                status = llm.check_llm()
        else:
            status = llm.check_llm()  # no network call, returns the setup hint
        st.session_state["llm_status"] = (key, status)
    return st.session_state["llm_status"][1]


def status_card(status: llm.LLMStatus) -> str:
    if status.ok:
        dot, title, msg = "ok", f"{status.provider} connected", ""
    elif config.llm_available():
        dot, title, msg = "bad", "Llama not reachable", status.message
    else:
        dot, title, msg = "warn", "No API key found", status.message
    model = status.model if config.llm_available() else config.DEFAULT_NEBIUS_MODEL
    body = f'<div class="status-msg">{h(msg)}</div>' if msg else ""
    return (f'<div class="status"><div class="status-head"><span class="dot {dot}"></span>{h(title)}</div>'
            f'<div class="status-model">{icon("cpu", 14)} {h(model)}</div>{body}</div>')


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-mark">{icon("file-text", 22)}</div>'
        f'<div><div class="brand-name">FilingSight</div><div class="brand-sub">Cited answers from 10-K filings</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Language model</div>', unsafe_allow_html=True)
    status = llm_status()
    st.markdown(status_card(status), unsafe_allow_html=True)
    with st.expander("Paste a key for this session", expanded=not config.llm_available()):
        typed = st.text_input("Nebius API key", type="password", key="runtime_key",
                              help="Kept in memory for this browser session only. For a permanent setup put it in .env.")
        if typed and typed != st.session_state.get("_applied_key"):
            config.set_runtime_key(typed)
            llm.reset()
            st.session_state["_applied_key"] = typed
            st.session_state.pop("llm_status", None)
            st.rerun()
    if config.llm_available() and st.button("Test connection", use_container_width=True):
        llm.reset()
        llm_status(force=True)
        st.rerun()

    llm_on = config.llm_available()
    use_llm = st.toggle("Write answers with Llama", value=llm_on and status.ok, disabled=not llm_on,
                        help="Off: answers are the top cited sentences (extractive), no LLM call.")

    st.markdown('<div class="side-label">Retrieval</div>', unsafe_allow_html=True)
    strategy = st.radio(
        "Chunking strategy", ["semantic", "fixed"], index=0 if config.DEFAULT_CHUNKING == "semantic" else 1,
        horizontal=True, help="semantic: embedding breakpoint chunks. fixed: 900 character recursive splitter.",
    )
    reranker = st.selectbox("Reranker", reranker_options(), help="local needs no download. none shows the raw hybrid order.")
    n_chunks = warm(strategy)

    st.markdown('<div class="side-label">Session</div>', unsafe_allow_html=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("history", None)

retr, ans = eval_summary()
best = max(retr, key=lambda r: r.get("hit@k", 0)) if retr else {}

# ---------------------------------------------------------------- nav + hero
REPO = "https://github.com/balum211/FilingSight"
mode = (f'<span class="dot ok"></span>{h(config.LLM_MODEL.split("/")[-1])}' if status.ok
        else '<span class="dot warn"></span>Extractive mode')
st.markdown(
    f'<nav class="topbar" aria-label="App"><span class="tb-brand">{icon("file-text", 18)}<span>FilingSight</span></span>'
    f'<span class="tb-pill" title="Language model status">{mode}</span>'
    f'<a class="tb-link" href="{REPO}" target="_blank" rel="noopener">{icon("github-logo", 16)} GitHub</a></nav>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""<div class="hero-grid"><div class="hero"><div class="eyebrow">{icon("book-open", 14)} Uber and Lyft · FY2021 Form 10-K</div>
<h1><span>Ask the annual reports.</span><span>Get answers you can cite.</span></h1>
<p>Hybrid search, reranking and a refusal gate over 275 filing pages,
so every number points back to the page it came from.</p></div>
<div class="preview" aria-label="Example answer"><div class="pv-q">How did Uber's revenue change in 2021?</div>
<div class="pv-a"><b>Uber:</b> Revenue increased &#36;6.3 billion, or 57%, primarily attributable to an increase in Gross Bookings of 56%. <span class="pv-cite">[1]</span></div>
<div class="pv-src">{icon("seal-check", 16)}<span><b>[1] Uber 10-K FY2021 · page 57</b> · Item 7 MD&amp;A</span></div>
<div class="meta" style="margin:0"><span class="chip">{icon("target", 12)} relevance 0.60 · gate 0.60</span><span class="chip">{icon("clock", 12)} about 4s</span></div>
</div></div>""",
    unsafe_allow_html=True,
)


def kpi(ic: str, label: str, value: str, note: str) -> str:
    return (f'<div class="kpi"><div class="kpi-top"><span>{h(label)}</span><span class="kpi-icon">{icon(ic, 16)}</span></div>'
            f'<div class="kpi-value">{h(value)}</div><div class="kpi-note">{h(note)}</div></div>')


st.markdown(
    '<div class="kpis">'
    + kpi("target", "Retrieval hit@5", f"{best.get('hit@k', 0):.2f}" if best else "n/a",
          f"{best.get('strategy', '')} · {best.get('mode', '')} · {best.get('reranker', '')} rerank" if best else "run the eval")
    + kpi("shield-check", "Unanswerable refused", f"{ans.get('refusal_accuracy', 0):.0%}" if ans else "n/a", "5 out of scope questions")
    + kpi("stack", "Chunks indexed", f"{n_chunks:,}", f"{strategy} chunking")
    + kpi("clock", "Median answer time", f"{ans.get('p50_latency_ms', 0) / 1000:.1f}s" if ans else "n/a", "budget 8s end to end")
    + "</div>",
    unsafe_allow_html=True,
)

@st.cache_data
def company_chunks(strategy: str) -> dict:
    counts: dict = {}
    for d in load_index(strategy).docs:
        c = d.metadata.get("company", "?")
        counts[c] = counts.get(c, 0) + 1
    return counts


cc = company_chunks(strategy)
CORPUS = [("uber", "Uber", "Uber Technologies, Inc.", "car-simple", 150, 307),
          ("lyft", "Lyft", "Lyft, Inc.", "car-simple", 125, 238)]
st.markdown(
    '<div class="corpus">' + "".join(
        f'<div class="co-card {k}"><div class="co-logo">{icon(ic, 24)}</div><div><div class="co-name">{name}</div>'
        f'<div class="co-meta">{h(full)} · Form 10-K FY2021</div></div><div class="co-stats">'
        f'<div class="co-stat"><div class="v">{body}</div><div class="l">body pages</div></div>'
        f'<div class="co-stat"><div class="v">{cc.get(name, 0):,}</div><div class="l">chunks</div></div></div></div>'
        for k, name, full, ic, body, _pdf in CORPUS) + "</div>",
    unsafe_allow_html=True,
)

chat_tab, eval_tab, how_tab = st.tabs(["Ask the filings", "Evaluation", "How it works"])


def render_sources(state: dict):
    sources = state.get("sources") or []
    if not sources:
        return
    cited = set(state.get("citations") or [])
    with st.expander(f"Sources · {len(sources)} retrieved, {len(cited)} cited"):
        cards = []
        top = max((float(s.get("score") or 0) for s in sources), default=1.0) or 1.0
        for s in sources:
            score = float(s.get("score") or 0)
            pct = max(4, min(100, int(score / top * 100)))
            pin = f'<span class="chip llm">{icon("quotes", 12)} cited</span>' if s["n"] in cited else ""
            cards.append(
                f'<div class="src {"cited" if s["n"] in cited else ""}"><div class="src-head">'
                f'<span class="src-n">[{s["n"]}]</span><span class="co {h(s["company"])}">{h(s["company"])}</span>'
                f'<span>10-K FY2021 · page {h(s["page"])}</span>{pin}'
                f'<span class="src-item">Item {h(s["item"])} {h(s["item_title"])}</span>'
                f'<span style="margin-left:auto" class="src-item mono">{score:.2f} <span class="bar"><span style="width:{pct}%"></span></span></span>'
                f'</div><div class="src-text">{h(s["text"][:1200])}</div></div>'
            )
        st.markdown("".join(cards), unsafe_allow_html=True)


def render_answer(state: dict):
    if state.get("llm_error"):
        st.markdown(
            f'<div class="notice warn">{icon("warning-circle", 18)}<div><b>Llama call failed, showing an extractive answer.</b> '
            f'{h(state["llm_error"])}</div></div>', unsafe_allow_html=True)
    if state["route"] == "refused":
        reason = f' <span style="color:var(--muted)">{h(state["refusal_reason"])}</span>' if state.get("refusal_reason") else ""
        st.markdown(f'<div class="notice bad">{icon("prohibit", 18)}<div><b>{h(state["answer"])}</b>{reason}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(md_safe(state["answer"]))

    t = state.get("timings_ms", {})
    chips = []
    gen = state.get("generator")
    if state["route"] == "refused":
        chips.append(f'<span class="chip ref">{icon("prohibit", 12)} refused</span>')
    elif gen == "llm":
        chips.append(f'<span class="chip llm">{icon("lightning", 12)} {h(config.LLM_MODEL.split("/")[-1])}</span>')
    elif gen:
        chips.append(f'<span class="chip ext">{icon("quotes", 12)} extractive</span>')
    chips.append(f'<span class="chip">{icon("clock", 12)} {t.get("total", 0) / 1000:.1f}s</span>')
    if state.get("top_score") is not None and state.get("sources"):
        chips.append(f'<span class="chip">{icon("target", 12)} relevance {state["top_score"]:.2f} · gate {state.get("threshold", 0):.2f}</span>')
    steps = " · ".join(f"{k} {v / 1000:.1f}s" for k, v in t.items() if k != "total")
    if steps:
        chips.append(f'<span class="chip">{icon("path", 12)} {h(steps)}</span>')
    st.markdown(f'<div class="meta">{"".join(chips)}</div>', unsafe_allow_html=True)
    render_sources(state)


with chat_tab:
    history = st.session_state.setdefault("history", [])
    if not history:
        st.markdown(
            f'<div class="empty"><div class="empty-icon">{icon("chat-circle-text", 24)}</div><div>'
            f'<h3>Start with a question about either filing</h3>'
            f'<p>Revenue, riders, risks or strategy. Pick a sample below or type your own. '
            f'Questions outside FY2019 to FY2021 are refused on purpose.</p></div></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, (tag, q) in enumerate(SAMPLES):
            with cols[i % 3]:
                st.markdown(f'<div class="sample-tag">{h(tag)}</div>', unsafe_allow_html=True)
                if st.button(q, key=f"sample{i}", use_container_width=True):
                    st.session_state["pending"] = q
                    st.rerun()
    for turn in history:
        with st.chat_message("user"):
            st.markdown(md_safe(turn["question"]))
        with st.chat_message("assistant", avatar="📑"):
            render_answer(turn["state"])

    question = st.chat_input("Ask about Uber or Lyft's 2021 annual report") or st.session_state.pop("pending", None)
    if question:
        skeleton = st.empty()
        skeleton.markdown(
            f'<div class="skel" aria-busy="true"><div class="skel-head">{icon("magnifying-glass", 14)} Retrieving, reranking and '
            f'answering: {h(question)}</div><div class="bar-s" style="width:92%"></div><div class="bar-s" style="width:78%"></div>'
            '<div class="bar-s" style="width:85%"></div><div class="bar-s" style="width:40%"></div></div>',
            unsafe_allow_html=True,
        )
        try:
            state = ask(question, strategy=strategy, reranker=reranker, use_llm=use_llm)
        except Exception as exc:  # surface provider errors instead of a stack trace
            skeleton.empty()
            st.markdown(f'<div class="notice bad">{icon("warning-circle", 18)}<div><b>The question could not be answered.</b> '
                        f'{h(llm.explain_error(exc))}</div></div>', unsafe_allow_html=True)
            st.stop()
        if state.get("llm_error"):
            st.session_state.pop("llm_status", None)  # recheck the badge on the next run
        history.append({"question": question, "state": state})
        st.rerun()  # redraw so the newest answer sits above the input box

with eval_tab:
    st.markdown(
        f'<div class="section-title">{icon("chart-bar", 20)} Chunking, retrieval and reranking compared</div>'
        '<p class="section-sub">25 labelled questions: 20 answerable, 5 unanswerable. Reproduce with '
        '<code>python eval/run_retrieval_eval.py</code> and <code>python eval/run_answer_eval.py</code>.</p>',
        unsafe_allow_html=True,
    )
    if retr:
        import altair as alt
        import pandas as pd

        df = pd.DataFrame([{"Configuration": f"{r['mode']} + {r['reranker']} rerank" if r["reranker"] != "none" else r["mode"],
                            "Chunking": r["strategy"], "Hit@5": r["hit@k"]} for r in retr])
        chart = (
            alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Configuration:N", sort=["dense", "bm25", "hybrid", "hybrid + local rerank"], title=None,
                        axis=alt.Axis(labelAngle=0)),
                xOffset="Chunking:N",
                y=alt.Y("Hit@5:Q", scale=alt.Scale(domain=[0, 1]), title="Hit@5"),
                color=alt.Color("Chunking:N", scale=alt.Scale(domain=["fixed", "semantic"], range=["#6B6B6B", "#7C9DFF"]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Chunking", "Configuration", alt.Tooltip("Hit@5:Q", format=".2f")],
            )
            .properties(height=280)
            .configure(background="transparent", font="Manrope")
            .configure_axis(gridColor="#313131", domainColor="#313131", tickColor="#313131", labelColor="#9B9B9B",
                            titleColor="#9B9B9B", labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor="#9B9B9B", labelFontSize=12)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True, theme=None)
        rows = []
        for r in retr:
            cls = ' class="best"' if r is best else ""
            rows.append(
                f"<tr{cls}><td>{h(r['strategy'])}</td><td>{h(r['mode'])}</td><td>{h(r['reranker'])}</td>"
                f"<td class='num'>{r['hit@k']:.2f}</td><td class='num'>{r['recall@k']:.2f}</td>"
                f"<td class='num'>{r['mrr@10']:.2f}</td><td class='num'>{r['precision@k']:.2f}</td>"
                f"<td class='num'>{r['p50_latency_ms']:,.0f} ms</td></tr>"
            )
        st.markdown(
            "<table class='tbl'><thead><tr><th>Chunking</th><th>Retrieval</th><th>Reranker</th><th class='num'>Hit@5</th>"
            "<th class='num'>Recall@5</th><th class='num'>MRR@10</th><th class='num'>Precision@5</th><th class='num'>p50</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>",
            unsafe_allow_html=True,
        )
    else:
        st.info("retrieval.json not generated yet. Run python eval/run_retrieval_eval.py.")
    for name, label in (("retrieval.md", "Full retrieval report"), ("answers.md", "Answer quality report")):
        path = RESULTS / name
        if path.exists():
            with st.expander(label):
                st.markdown(md_safe(path.read_text(encoding="utf-8")))

with how_tab:
    st.markdown(
        f'<div class="section-title">{icon("flow-arrow", 20)} The answer flow</div>'
        '<p class="section-sub">A LangGraph state machine. The refusal path was designed first, because a wrong number '
        'in a financial answer is worse than saying the filings do not cover it.</p>',
        unsafe_allow_html=True,
    )
    STEPS = [
        ("funnel", "Analyze", "Detects company, year and question type, and splits comparisons into one query per company.",
         "Year outside 2019 to 2021: refuse"),
        ("magnifying-glass", "Retrieve", "Chroma dense search plus BM25 keywords, fused with weighted RRF, 20 candidates per company.", ""),
        ("arrows-clockwise", "Rerank", "Local sentence and key term reranker (or cross encoder, or Llama) picks a company balanced top 5.", ""),
        ("shield-check", "Gate", "Checks the best reranked score against the confidence threshold before any answer is written.",
         "Low relevance: refuse"),
        ("lightning", "Generate", "Llama 3.3 70B writes from numbered sources. If it is unavailable the top cited sentences are shown.", ""),
        ("check-circle", "Validate", "Strips citations that point nowhere and retries once if none are valid.", "NOT_FOUND from the model: refuse"),
    ]
    cards = []
    for i, (ic, title, text, exit_) in enumerate(STEPS, 1):
        ex = f'<div class="exit">{icon("prohibit", 12)} {h(exit_)}</div>' if exit_ else ""
        cards.append(f'<div class="step"><div class="step-top"><div class="step-icon">{icon(ic, 20)}</div>'
                     f'<div><div class="step-n">0{i}</div><h4>{h(title)}</h4></div></div><p>{h(text)}</p>{ex}</div>')
    st.markdown(f'<div class="steps">{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-title">{icon("database", 20)} Design decisions</div>', unsafe_allow_html=True)
    DECISIONS = [
        ("Corpus", "Uber and Lyft 10-K for FY2021 (SEC filings, PDF, English), 275 body pages after dropping exhibits"),
        ("Cleaning", "PyMuPDF text, page numbers dropped, table cells joined into rows, wrapped lines reflowed, Item tagged"),
        ("Chunking", "Fixed 900/150 characters vs semantic (embedding breakpoints, 85th percentile, up to 1,100 characters)"),
        ("Embedding", f"{EMBEDDING_MODEL_NAME}, 384 dimensions, ONNX on CPU, chunk size matched to its 256 token window"),
        ("Retrieve", f"Chroma (cosine) plus BM25, weighted RRF, metadata filter by company, top {config.TOP_K}"),
        ("Refusal", "Year scope rule, confidence gate, NOT_FOUND from the LLM, citation validation"),
    ]
    rows = "".join(f"<tr><td style='width:160px;color:var(--muted);font-weight:600'>{h(k)}</td><td>{h(v)}</td></tr>" for k, v in DECISIONS)
    st.markdown(f"<table class='tbl'><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

# ---------------------------------------------------------------- tagline + footer
TAGLINE_LINES = ["Retrieval decided quality, not the model.", "Hit@5 went from 0.65 to 0.95", "without touching the LLM."]
st.markdown(
    '<section class="tagline" aria-label="Key finding"><p>'
    + "<br>".join(" ".join(f'<span class="w" style="--i:{i + 8 * n}">{h(w)}</span>' for i, w in enumerate(line.split()))
                  for n, line in enumerate(TAGLINE_LINES))
    + "</p></section>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<footer class="foot"><span>For learning purposes only, not investment advice. Figures come from FY2021 filings.</span>'
    f'<span class="links"><a href="{REPO}" target="_blank" rel="noopener">{icon("github-logo", 14)} Source code</a>'
    f'<a href="https://www.sec.gov/edgar/search/" target="_blank" rel="noopener">{icon("arrow-up-right", 14)} SEC EDGAR</a>'
    f'<a href="https://github.com/elayadesign/ai-design-skills" target="_blank" rel="noopener">{icon("sparkle", 14)} Design rules</a></span></footer>',
    unsafe_allow_html=True,
)
