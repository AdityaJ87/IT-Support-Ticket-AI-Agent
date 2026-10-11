# 🤖 IT Support Ticket AI Agent

An AI-powered IT support ticket classification system combining **Machine Learning**, **historical ticket retrieval**, and the **Gemini API** to classify support requests and recommend next steps.

## Overview

The application accepts a ticket written in natural language and produces eight structured fields:

- **Issue type** — Technical or Non-Technical
- **Department** — Likely responsible department
- **Urgency** — High, Medium, or Low
- **Findings** — Technical, Billing, Marketing, Sales, or Support
- **Severity** — Low, Medium, High, Highest, or Critical
- **Reason** — Concise explanation of the classification
- **Recommended action** — Suggested next step
- **Confidence** — High, Medium, or Low

## How It Works

1. **Load historical data:** Pandas reads the Excel dataset.
2. **Predict a department:** A Scikit-learn pipeline uses TF-IDF and Logistic Regression trained on historical ticket text and department labels.
3. **Retrieve similar tickets:** TF-IDF vectors and cosine similarity identify up to five similar historical tickets.
4. **Analyze with Gemini:** The application sends the new ticket, ML prediction, and retrieved examples to the Gemini API with classification instructions.
5. **Display results:** The Gemini response is parsed as JSON and displayed in Streamlit.

> **Note:** Gemini is called through its API; this project does not fine-tune Gemini. The local Logistic Regression model is trained on the labeled dataset.

## Architecture

```text
User enters ticket
       |
       v
Streamlit interface
       |
       v
Ticket text
   |                 |
   v                 v
TF-IDF +             TF-IDF retrieval
Logistic Regression  + cosine similarity
   |                 |
   +--------+--------+
            |
            v
        Gemini API
            |
            v
       JSON response
            |
            v
 Eight classification fields
```

## Tech Stack

- **Python**
- **Streamlit** — interactive web interface
- **Pandas** and **openpyxl** — Excel loading and processing
- **Scikit-learn** — TF-IDF, Logistic Regression, and cosine similarity
- **Google Gen AI SDK** (`google-genai`) — Gemini API integration
- **python-dotenv** — environment variable configuration

## Project Structure

```text
IT-Support-Ticket-AI-Agent/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── tickets.xlsx   # Local dataset; publish only if permitted
```

The app expects `data/tickets.xlsx` with these columns:

- `Ticket_ID`
- `Subject`
- `Description`
- `Department`

Additional columns may be present.

## Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/AdityaJ87/IT-Support-Ticket-AI-Agent.git
cd IT-Support-Ticket-AI-Agent
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_available_gemini_model
```


### 5. Add the dataset

Place the permitted Excel dataset at:

```text
data/tickets.xlsx
```

The dataset is not included by default. If it was supplied by hackathon organizers, confirm permission before sharing it publicly.

### 6. Run Streamlit

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Example Inputs

Try these sample tickets:

1. `A production server is down and multiple employees cannot access the application.`
2. `An employee cannot log in after resetting their password.`
3. `A cloud services invoice contains an incorrect charge.`
4. `A user reports a suspicious login from an unknown location.`

The department result depends on the labels and examples in the dataset.

## Output Format

The application returns a JSON-derived result with these fields:

```json
{
  "issue_type": "Technical or Non-Technical",
  "department": "Responsible department",
  "urgency": "High, Medium, or Low",
  "findings": "Technical, Billing, Marketing, Sales, or Support",
  "severity": "Low, Medium, High, Highest, or Critical",
  "reason": "One concise sentence explaining the classification",
  "recommended_action": "One specific next action",
  "confidence": "High, Medium, or Low"
}
```

These are model-generated classifications, not guaranteed ground truth.

## Security Notes

- Keep API credentials in `.env`; never hard-code keys in source files.
- Keep `.env` in `.gitignore`.
- Do not publish private, confidential, or organizer-provided datasets without permission.
- Do not use real employee or customer information in public examples.

## Limitations

- Classification quality depends on the size, quality, and consistency of the historical dataset.
- Text similarity does not prove two tickets have the same root cause.
- Logistic Regression's predicted probability is not necessarily calibrated.
- Gemini's confidence field is a qualitative model assessment, not a measured accuracy score.
- The app suggests actions; it does not automatically assign tickets, execute remediation, or verify resolution.
- A working Gemini API key, network access, and an available model are required.

## Future Improvements

- Evaluate the classifier with a held-out test set and per-class precision, recall, and F1-score.
- Add human review for security incidents and high-impact outages.
- Add structured logging, API retries, and an ML-only fallback.
- Integrate with a ticketing platform for controlled assignment and status updates.
- Add authentication and role-based access for organizational deployment.

