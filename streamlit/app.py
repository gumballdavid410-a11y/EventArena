import os, re, sqlite3
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
DATA = HERE / "eventarena_synthetic_intents.csv"
MODEL = HERE / "intent_classifier.joblib"
THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.30"))

EVENTS = [
("UEFA Champions League: Madrid vs Manchester","Sports","2026-11-15","21:00","Santiago Bernabéu Stadium",120.00),
("Rock Legends World Tour","Concerts","2026-11-20","19:30","Arena Metropolis",85.00),
("Neon Horizon Electronic Music Festival","Festivals","2026-12-05","16:00","Sunset Beach Park",45.00),
("Grand Slam Tennis Finals","Sports","2026-11-18","14:00","National Tennis Arena",150.00),
("Acoustic Evening with Symphony Strings","Concerts","2026-12-12","20:00","Royal Concert Hall",35.00),
]

def get_model():
    if MODEL.exists():
        return joblib.load(MODEL)
    data = pd.read_csv(DATA)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])
    pipe.fit(data["text"], data["intent"])
    joblib.dump(pipe, MODEL)
    return pipe

@st.cache_resource
def cached_model():
    return get_model()

def rule_intent(text):
    if re.search(r"\b(hi|hello|hey|greetings|good morning)\b", text, re.I):
        return "greeting"
    if re.search(r"\b(my name is|i am|i'm|call me)\b", text, re.I):
        return "name"
    return None

def events_reply(intent):
    if intent == "cheap":
        rows = [e for e in EVENTS if e[5] <= 50]
        heading = "Affordable events (up to $50)"
    else:
        cat = {"sport":"Sports","concert":"Concerts","festival":"Festivals"}[intent]
        rows = [e for e in EVENTS if e[1] == cat]
        heading = cat + " events"
    if not rows:
        return "No matching events are listed right now."
    return "### " + heading + "\n" + "\n".join(
        f"- **{e[0]}** — {e[2]} at {e[3]}, {e[4]} — **${e[5]:.2f}**" for e in rows
    )

def respond(text, model):
    simple = rule_intent(text)
    if simple == "greeting":
        return "Hello! Welcome to EventArena. What kind of event would you like to discover?", "rule: greeting", None
    if simple == "name":
        m = re.search(r"\b(?:my name is|i am|i'm|call me)\s+([A-Za-z'-]+)", text, re.I)
        if m:
            st.session_state.visitor_name = m.group(1)
            return f"Nice to meet you, {m.group(1)}!", "rule: name", None
        return "Tell me your name using 'My name is Alex'.", "rule: name", None
    probs = model.predict_proba([text])[0]
    idx = int(probs.argmax())
    intent, confidence = str(model.classes_[idx]), float(probs[idx])
    if intent == "out_of_scope" or confidence < THRESHOLD:
        return "🤔 I couldn't confidently identify that request. I can help with sports, concerts, festivals and affordable events.", "fallback: " + intent, confidence
    if intent in ("sport","concert","festival","cheap"):
        return events_reply(intent), "classifier: " + intent, confidence
    if intent == "help":
        return "I can find sports matches, concerts, festivals and affordable events costing $50 or less.", "classifier: help", confidence
    if intent == "exit":
        st.session_state.awaiting_exit = True
        return "Are you sure you want to end this chat? Reply yes or no.", "classifier: exit", confidence
    return "I couldn't handle that request. Try asking about an event.", "fallback: unhandled intent", confidence

st.set_page_config(page_title="EventArena", page_icon="🎟️")
st.title("🎟️ EventArena")
st.caption("Event discovery chatbot — synthetic training data prototype")
try:
    model = cached_model()
except Exception as exc:
    st.error(f"Could not load/train the classifier: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Welcome to EventArena! Ask me about sports, concerts, festivals or affordable events."}]
if "awaiting_exit" not in st.session_state:
    st.session_state.awaiting_exit = False
with st.sidebar:
    st.write(f"Confidence threshold: **{THRESHOLD:.2f}**")
    st.success("Intent classifier loaded")
    if st.button("Start a new conversation"):
        for key in ("messages","awaiting_exit","visitor_name"):
            st.session_state.pop(key, None)
        st.rerun()
    st.caption("Use a second browser/private window to demonstrate an independent chat.")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
prompt = st.chat_input("Message EventArena")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    if st.session_state.awaiting_exit:
        answer = "Goodbye! Thanks for using EventArena." if re.search(r"\b(yes|confirm)\b",prompt,re.I) else "No problem. Let's continue browsing."
        st.session_state.awaiting_exit = False
        route, confidence = "rule: exit confirmation", None
    else:
        answer, route, confidence = respond(prompt, model)
    st.session_state.messages.append({"role":"assistant","content":answer})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Routing details"):
            st.write(route)
            if confidence is not None:
                st.write(f"Confidence: {confidence:.3f} | Threshold: {THRESHOLD:.3f}")
    st.rerun()
