# app.py
# Kart AI v2 — Streamlit UI
#
# Satisfies:
#   FR5  - Conversational Ability: full chat UI with streaming + memory
#   NFR2 - Local Deployability: runs entirely on localhost via Streamlit

import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import streamlit as st

# ── Shutdown server (port 7861) ────────────────────────────────────────────────
# A minimal HTTP server that kills the process when the browser tab closes.
# The tab sends navigator.sendBeacon to this endpoint on beforeunload.
_SHUTDOWN_PORT = 7861

class _ShutdownHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        threading.Thread(
            target=lambda: (time.sleep(0.4), os._exit(0)), daemon=True
        ).start()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def log_message(self, *_):
        pass  # suppress console noise

def _start_shutdown_server():
    try:
        srv = HTTPServer(("127.0.0.1", _SHUTDOWN_PORT), _ShutdownHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    except OSError:
        pass  # already running (Streamlit reruns the script on every interaction)

_start_shutdown_server()

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Kart",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── Design tokens ─────────────────────────────────────────────────── */
:root {
    --bg-page:      #f4f4f5;
    --bg-stage:     #ffffff;
    --surface:      #f0f0f0;
    --surface2:     #f5f5f5;
    --surface3:     #e8e8e8;
    --border:       #e2e2e2;
    --border-2:     #cccccc;
    --border-hi:    #aaaaaa;
    --accent:       #111111;
    --accent-dim:   rgba(0,0,0,0.05);
    --text:         #111111;
    --text-2:       #666666;
    --text-3:       #aaaaaa;
    --radius-pill:  99px;
    --radius-md:    14px;
    --radius-sm:    9px;
    --shadow-float: 0 4px 24px rgba(0,0,0,0.09), 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Global ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    background-color: var(--bg-page) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp { background: var(--bg-page) !important; }

#MainMenu, footer, header,
.stDeployButton, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* Chat stage — truly centered, framed with side borders */
.main > div { display: flex !important; justify-content: center !important; }
.block-container {
    max-width: 800px !important;
    width: 100% !important;
    padding: 0 40px 130px !important;
    margin: 0 auto !important;
    background: var(--bg-stage) !important;
    border-left:  1px solid var(--border) !important;
    border-right: 1px solid var(--border) !important;
    min-height: 100vh !important;
    box-sizing: border-box !important;
}

/* ── Animations ──────────────────────────────────────────────────────── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1;   transform: scale(1); }
    50%       { opacity: 0.35; transform: scale(0.55); }
}
@keyframes typingBounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.25; }
    40%            { transform: translateY(-5px); opacity: 1; }
}

/* ── Nav bar ─────────────────────────────────────────────────────────── */
.kart-nav-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.75rem;
    letter-spacing: -0.5px;
    color: #111111;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 24px 0 16px;
}

.kart-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2.8s ease-in-out infinite;
}

.kart-divider {
    height: 1px;
    background: var(--border);
    margin: 0 0 32px;
}

/* Nav icon buttons */
.nav-actions .stButton > button {
    padding: 4px 10px !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.82rem !important;
    border-color: var(--border) !important;
    margin-top: 18px !important;
}

/* ── User message ────────────────────────────────────────────────────── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 16px;
    animation: fadeSlideUp 0.28s cubic-bezier(0.16,1,0.3,1) forwards;
}

.msg-user-bubble {
    background: #f0f0f0;
    border: 1px solid #d8d8d8;
    border-radius: 20px 20px 4px 20px;
    padding: 12px 18px;
    max-width: 70%;
    font-size: 0.93rem;
    line-height: 1.7;
    color: #111111;
    word-wrap: break-word;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Assistant message ───────────────────────────────────────────────── */
.msg-assistant {
    display: flex;
    gap: 14px;
    margin-bottom: 32px;
    margin-top: 8px;
    align-items: flex-start;
    animation: fadeSlideUp 0.28s cubic-bezier(0.16,1,0.3,1) forwards;
}

.msg-avatar {
    width: 32px; height: 32px;
    background: var(--accent);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.04), 0 2px 8px rgba(0,0,0,0.15);
}

.msg-body { flex: 1; min-width: 0; }

.msg-content {
    font-size: 0.93rem;
    line-height: 1.85;
    color: #111111;
    word-wrap: break-word;
}

.msg-content strong { color: #111111; font-weight: 600; }
.msg-content em     { color: var(--text-2); font-style: normal; font-size: 0.83rem; }
.msg-content ul, .msg-content ol { padding-left: 20px; margin: 7px 0; }
.msg-content li { margin: 4px 0; }
.msg-content p  { margin: 5px 0; }

/* ── Agent tag ───────────────────────────────────────────────────────── */
.agent-tag {
    display: inline-block;
    margin-top: 10px;
    font-size: 0.66rem;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-3);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 2px 10px;
}
.agent-tag.search  { border-color: rgba(0,0,0,0.12); color: rgba(0,0,0,0.38); }
.agent-tag.analyst { border-color: rgba(37,99,235,0.2);  color: rgba(37,99,235,0.65); }

/* ── Typing dots ─────────────────────────────────────────────────────── */
.typing-dot {
    display: inline-block;
    width: 5px; height: 5px;
    background: var(--text-3);
    border-radius: 50%;
    animation: typingBounce 1.1s ease-in-out infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.18s; }
.typing-dot:nth-child(3) { animation-delay: 0.36s; }
.typing-wrap { display: flex; gap: 5px; align-items: center; padding: 8px 0; }

/* ── Example chips ───────────────────────────────────────────────────── */
.chips-section { margin: 32px 0 16px; text-align: center; }
.chips-label {
    font-size: 0.66rem;
    font-weight: 500;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 14px;
}

/* ── All Streamlit buttons ───────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(0,0,0,0.14) !important;
    border-radius: 999px !important;
    color: #555555 !important;
    font-size: 0.77rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 6px 16px !important;
    transition: border-color 0.18s ease, color 0.18s ease,
                background 0.18s ease, transform 0.18s ease,
                box-shadow 0.18s ease !important;
    white-space: normal !important;
    word-break: break-word !important;
}
.stButton > button:hover {
    border-color: rgba(0,0,0,0.32) !important;
    color: #111111 !important;
    background: rgba(0,0,0,0.04) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    opacity: 0.22 !important;
    cursor: not-allowed !important;
}

/* ── Chat input — pill ───────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 28px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: calc(100% - 48px) !important;
    max-width: 720px !important;
    padding: 0 !important;
    background: transparent !important;
    z-index: 999 !important;
}

[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.14) !important;
    border-radius: 999px !important;
    color: #111111 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 13px 52px 13px 24px !important;
    caret-color: var(--accent) !important;
    box-shadow: var(--shadow-float) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(0,0,0,0.32) !important;
    box-shadow: var(--shadow-float), 0 0 0 2px rgba(0,0,0,0.05) !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-3) !important; }

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] button,
[data-testid="stChatInput"] button:focus,
[data-testid="stChatInput"] button:active,
[data-testid="stChatInput"] button:focus-visible {
    background: var(--accent) !important;
    border-radius: 50% !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    transition: opacity 0.15s ease, transform 0.15s ease !important;
}
[data-testid="stChatInput"] button:hover {
    opacity: 0.82 !important;
    transform: scale(0.94) !important;
    box-shadow: none !important;
}

/* ── Disclaimer ──────────────────────────────────────────────────────── */
.kart-disclaimer {
    position: fixed;
    bottom: 9px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.66rem;
    color: var(--text-3);
    white-space: nowrap;
    z-index: 1000;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.1px;
}

/* ── Misc ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: var(--text-2) !important; }
.stAlert { background: var(--surface) !important; border-color: var(--border) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Shutdown beacon JS ────────────────────────────────────────────────────────
st.markdown(f"""
<script>
window.addEventListener('beforeunload', function () {{
    navigator.sendBeacon('http://127.0.0.1:{_SHUTDOWN_PORT}/');
}});
</script>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────
st.markdown('<div class="kart-disclaimer">Kart is an AI and can make mistakes.</div>', unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # list of {"role": ..., "content": ..., "intent": ...}
if "ingested" not in st.session_state:
    st.session_state.ingested = False

# ── Lazy import router (so UI loads even if ingestion not done) ────────────
@st.cache_resource(show_spinner=False)
def load_backend():
    """Import router once and cache. Also triggers KB check."""
    from knowledge_base import get_collection
    from router import route_stream
    col = get_collection()
    count = col.count()
    return route_stream, count

# ── Avatar SVG (shopping bag icon, dark stroke on white bg) ───────────────
_AVATAR = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" '
    'fill="none" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M6 2L3 6v14a2 2 0 002 2h14a2 2 0 002-2V6l-3-4z"/>'
    '<line x1="3" y1="6" x2="21" y2="6"/>'
    '<path d="M16 10a4 4 0 01-8 0"/>'
    '</svg>'
)

# ── Header nav: logo left, action icons right ─────────────────────────────
hcol_logo, _, hcol_act = st.columns([4, 7, 3])
with hcol_logo:
    st.markdown('<div class="kart-nav-logo">Kart <span class="kart-dot"></span></div>', unsafe_allow_html=True)
with hcol_act:
    st.markdown('<div class="nav-actions">', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.button("◷", key="btn_hist", help="History", disabled=True)
    with a2:
        if st.button("+", key="btn_new", help="New Chat"):
            st.session_state.messages = []
            st.rerun()
    with a3:
        if st.button("✕", key="btn_clr", help="Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="kart-divider"></div>', unsafe_allow_html=True)

# ── Load backend + show KB status ──────────────────────────────────────────
try:
    route_stream, product_count = load_backend()
    backend_ready = True
except Exception as e:
    backend_ready = False
    st.error(f"⚠️ Backend not ready: {e}. Make sure Ollama is running and ingestion is complete.")

# ── Example chips (only shown when chat is empty) ──────────────────────────
EXAMPLES = [
    "Sports footwear under ₹2000",
    "Cheapest 5 watches",
    "Average jewellery price",
    "Top rated home decor",
    "Top clothing brands",
    "Beauty gifts under ₹500",
]

if not st.session_state.messages:
    st.markdown("""
    <div class="chips-section">
        <div class="chips-label">Try asking</div>
        <div class="chips-grid">
    """, unsafe_allow_html=True)

    row1 = st.columns(3)
    row2 = st.columns(3)
    all_cols = row1 + row2
    for i, ex in enumerate(EXAMPLES):
        with all_cols[i]:
            if st.button(ex, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_input = ex
                st.rerun()


# ── Render conversation history ────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    """Convert common LLM markdown to HTML for rendering inside custom divs."""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Bullet lists
    lines = text.split('\n')
    out, in_ul = [], False
    for line in lines:
        s = line.strip()
        if s.startswith(('- ', '* ', '• ')):
            if not in_ul:
                out.append('<ul style="padding-left:18px;margin:6px 0">')
                in_ul = True
            out.append(f'<li>{s[2:].strip()}</li>')
        else:
            if in_ul:
                out.append('</ul>')
                in_ul = False
            out.append(f'<p style="margin:4px 0">{s}</p>' if s else '')
    if in_ul:
        out.append('</ul>')
    return '\n'.join(out)


def render_message(msg: dict):
    role    = msg["role"]
    content = msg["content"]
    intent  = msg.get("intent", "")

    if role == "user":
        # No indentation on closing tags — 4-space indent = markdown code block
        st.markdown(
            f'<div class="msg-user"><div class="msg-user-bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        tag_html = ""
        if intent == "search":
            tag_html = '<span class="agent-tag search">Search Agent</span>'
        elif intent == "analytical":
            tag_html = '<span class="agent-tag analyst">Analyst Agent</span>'

        st.markdown(
            f'<div class="msg-assistant">'
            f'<div class="msg-avatar">{_AVATAR}</div>'
            f'<div class="msg-body">'
            f'<div class="msg-content">{_md_to_html(content)}</div>'
            f'{tag_html}'
            f'</div></div>',
            unsafe_allow_html=True,
        )


for msg in st.session_state.messages:
    render_message(msg)

# ── Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input(
    placeholder="Ask about products, prices, comparisons…",
    disabled=not backend_ready,
)

# Handle chip click (pending input)
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

# ── Process input ──────────────────────────────────────────────────────────
if user_input and user_input.strip() and backend_ready:
    query = user_input.strip()

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "intent": ""
    })
    render_message(st.session_state.messages[-1])

    # Show typing indicator while processing
    typing_placeholder = st.empty()
    typing_placeholder.markdown(f"""
    <div class="msg-assistant">
        <div class="msg-avatar">{_AVATAR}</div>
        <div class="msg-body">
            <div class="typing-wrap">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Build history for router (exclude last user message — already appended)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # Stream response — typing dots stay until first chunk arrives
    accumulated = ""
    intent = ""
    try:
        for chunk, intent_chunk in route_stream(query, history):
            if not accumulated:
                typing_placeholder.empty()  # drop dots on first chunk
            intent = intent_chunk
            accumulated += chunk
            typing_placeholder.markdown(
                f'<div class="msg-assistant">'
                f'<div class="msg-avatar">{_AVATAR}</div>'
                f'<div class="msg-body">'
                f'<div class="msg-content">{accumulated}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        accumulated = f"Sorry, I ran into an error: {e}"
        intent = ""

    typing_placeholder.empty()

    # Save and re-render with agent tag
    st.session_state.messages.append({
        "role":    "assistant",
        "content": accumulated,
        "intent":  intent,
    })
    render_message(st.session_state.messages[-1])

    # Rerun to refresh clear button visibility
    st.rerun()

