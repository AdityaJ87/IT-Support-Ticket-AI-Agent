import os
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DATA_FILE = os.path.join(BASE_DIR, "data", "tickets.xlsx")

OUTPUT_FIELDS = [
    "issue_type",
    "department",
    "urgency",
    "findings",
    "severity",
    "reason",
    "recommended_action",
    "confidence",
]

st.set_page_config(
    page_title="IT Support Ticket AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 IT Support Ticket AI Agent")
st.caption("Machine Learning + Historical Ticket Retrieval + Gemini AI")


# =========================================================
# 2. LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset(path):
    data = pd.read_excel(path)
    data.columns = data.columns.astype(str).str.strip()
    return data.fillna("")


if not os.path.exists(DATA_FILE):
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info("Place tickets.xlsx inside the Code/data folder.")
    st.stop()

try:
    df = load_dataset(DATA_FILE)
except Exception as exc:
    st.error(f"Unable to read the Excel dataset: {exc}")
    st.stop()

required_columns = ["Ticket_ID", "Subject", "Description", "Department"]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

for column in required_columns:
    df[column] = df[column].astype(str).str.strip()

df["ticket_text"] = (df["Subject"] + " " + df["Description"]).str.strip()

training_df = df[
    (df["ticket_text"] != "")
    & (df["Department"] != "")
    & (~df["Department"].str.lower().isin(["nan", "none", "unknown"]))
].copy()


# =========================================================
# 3. TRAIN DEPARTMENT CLASSIFIER
# =========================================================

@st.cache_resource
def train_classifier(texts, labels):
    classifier = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])
    classifier.fit(texts, labels)
    return classifier


ml_model = None
training_error = None

if len(training_df) >= 2 and training_df["Department"].nunique() >= 2:
    try:
        ml_model = train_classifier(
            training_df["ticket_text"].tolist(),
            training_df["Department"].tolist(),
        )
    except Exception as exc:
        training_error = str(exc)


# =========================================================
# 4. BUILD HISTORICAL TICKET RETRIEVAL INDEX
# =========================================================

@st.cache_resource
def build_retrieval_index(texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        max_features=10000,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


retrieval_index = None
try:
    valid_texts = df["ticket_text"].tolist()
    if any(text.strip() for text in valid_texts):
        retrieval_index = build_retrieval_index(valid_texts)
except ValueError:
    retrieval_index = None


def retrieve_tickets(question, limit=5):
    """Return the most textually similar historical tickets."""
    if retrieval_index is None:
        return []

    vectorizer, historical_matrix = retrieval_index

    try:
        question_vector = vectorizer.transform([question])
        similarities = cosine_similarity(
            question_vector, historical_matrix
        ).flatten()

        ranked_indices = similarities.argsort()[::-1]
        results = []

        for index in ranked_indices:
            score = float(similarities[index])
            if score <= 0:
                break

            row = df.iloc[index]
            results.append({
                "Ticket_ID": str(row["Ticket_ID"]),
                "Subject": str(row["Subject"]),
                "Description": str(row["Description"])[:500],
                "Department": str(row["Department"]),
                "similarity": round(score, 3),
            })

            if len(results) >= limit:
                break

        return results
    except Exception:
        return []


# =========================================================
# 5. ML DEPARTMENT PREDICTION
# =========================================================

def predict_department(question):
    if ml_model is None:
        return {
            "department": "Insufficient information",
            "confidence": None,
            "note": training_error
            or "The dataset needs at least two department classes.",
        }

    probabilities = ml_model.predict_proba([question])[0]
    best_index = probabilities.argmax()

    return {
        "department": str(ml_model.classes_[best_index]),
        "confidence": round(float(probabilities[best_index]), 3),
        "note": "Prediction from the TF-IDF classifier.",
    }


# =========================================================
# 6. GEMINI TICKET ANALYSIS
# =========================================================

def analyze_ticket(question):
    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing from the Code/.env file."
        )

    examples = retrieve_tickets(question)
    prediction = predict_department(question)
    client = genai.Client(api_key=API_KEY)

    department_labels = sorted(
        training_df["Department"].dropna().unique().tolist()
    )

    prompt = f"""
You are an IT Support Ticket Classification AI Agent.

Analyze the user's ticket using the ticket text, historical examples,
and ML department prediction provided below.

NEW TICKET:
{question}

AVAILABLE DEPARTMENT LABELS:
{json.dumps(department_labels, ensure_ascii=False)}

ML DEPARTMENT PREDICTION:
{json.dumps(prediction, ensure_ascii=False)}

SIMILAR HISTORICAL TICKETS:
{json.dumps(examples, ensure_ascii=False)}

Return ONLY a valid JSON object with exactly these eight fields:

{{
  "issue_type": "Technical or Non-Technical",
  "department": "Responsible department",
  "urgency": "High, Medium, or Low",
  "findings": "Technical, Billing, Marketing, Sales, or Support",
  "severity": "Low, Medium, High, Highest, or Critical",
  "reason": "One concise sentence explaining the classification",
  "recommended_action": "One specific next action",
  "confidence": "High, Medium, or Low"
}}

CLASSIFICATION RULES:
1. Use actual department labels from the dataset when appropriate.
2. Select the responsible team, not a device or technology name.
3. issue_type must be Technical or Non-Technical.
4. findings must be one of Technical, Billing, Marketing, Sales, or Support.
5. urgency depends on time sensitivity, service disruption, and impact.
6. severity depends on actual incident impact and criticality.
   Do not assign severity merely from the department or issue category.
7. High urgency and high severity are not automatically equivalent.
8. Use historical examples as evidence, not unquestionable truth.
9. Never invent facts, affected-user counts, or business impact.
10. If essential information is missing, state "Insufficient information"
    in the relevant field or explain the limitation in reason.
11. confidence must be High, Medium, or Low and reflect available evidence.
12. Keep every field concise and specific.
13. Do not return additional fields, Markdown, or explanatory text.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    result = json.loads(response.text)
    if not isinstance(result, dict):
        raise ValueError("Gemini did not return a JSON object.")

    cleaned_result = {}
    for field in OUTPUT_FIELDS:
        value = result.get(field, "Insufficient information")
        if value is None or str(value).strip() == "":
            value = "Insufficient information"
        cleaned_result[field] = value

    return {
        "result": cleaned_result,
        "evidence": examples,
        "ml_prediction": prediction,
    }


# =========================================================
# 7. DISPLAY FUNCTION — MUST BE DEFINED BEFORE IT IS CALLED
# =========================================================

def display_result(result):
    """Display exactly the eight required output fields."""
    first_row = st.columns(3)

    first_row[0].metric("Issue Type", result.get("issue_type", "Unknown"))
    first_row[1].metric("Urgency", result.get("urgency", "Unknown"))
    first_row[2].metric("Severity", result.get("severity", "Unknown"))

    st.write("**Department**")
    st.write(result.get("department", "Unknown"))

    st.write("**Findings**")
    st.write(result.get("findings", "Unknown"))

    st.write("**Reason**")
    st.write(result.get("reason", "Unknown"))

    st.write("**Recommended Action**")
    st.write(result.get("recommended_action", "Unknown"))

    st.write("**Confidence**")
    st.write(result.get("confidence", "Unknown"))


# =========================================================
# 8. SIDEBAR
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge Base")
    st.metric("Historical Tickets", len(df))
    st.metric("Training Records", len(training_df))
    st.metric("Departments Learned", training_df["Department"].nunique())

    st.write("**ML Classifier**")
    st.write(
        "Trained" if ml_model is not None
        else "Unavailable — check the department labels."
    )

    st.write("**Gemini Model**")
    st.code(MODEL)

    if st.button("Preview Dataset", use_container_width=True):
        st.dataframe(df[required_columns].head(10), use_container_width=True)

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# 9. CHAT HISTORY
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(
            message.get("result"), dict
        ):
            display_result(message["result"])
        else:
            st.markdown(message["content"])


# =========================================================
# 10. SAMPLE TICKETS
# =========================================================

st.markdown("### Try a sample ticket")

samples = [
    "A production server is down and multiple employees cannot access the application.",
    "An employee cannot log in after resetting their password.",
    "A cloud services invoice contains an incorrect charge.",
    "A user reports a suspicious login from an unknown location.",
]

columns = st.columns(2)
for index, sample in enumerate(samples):
    with columns[index % 2]:
        if st.button(
            f"Example {index + 1}",
            key=f"sample_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = sample
            st.rerun()


# =========================================================
# 11. CHAT INPUT AND ANALYSIS
# =========================================================

question = st.chat_input("Describe an IT support ticket...")

if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Checking historical tickets and consulting Gemini..."
        ):
            try:
                analysis = analyze_ticket(question)
                result = analysis["result"]

                display_result(result)

                with st.expander("View supporting evidence"):
                    st.write("**ML Department Prediction**")
                    st.json(analysis["ml_prediction"])

                    st.write("**Similar Historical Tickets**")
                    if analysis["evidence"]:
                        st.dataframe(
                            pd.DataFrame(analysis["evidence"]),
                            use_container_width=True,
                        )
                    else:
                        st.info(
                            "No sufficiently similar historical tickets found."
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Ticket analysis completed.",
                    "result": result,
                })

            except json.JSONDecodeError:
                st.error("Gemini returned invalid JSON. Please try again.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.info(
                    "Check the API key, model name, internet connection, "
                    "API quota, and dataset."
                )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()
st.caption(
    "IT Support AI Agent | TF-IDF + Logistic Regression "
    "+ Historical Retrieval + Gemini API"
)
