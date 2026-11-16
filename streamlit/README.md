# EventArena — Portfolio Project 2 Part 2 (Streamlit)

This folder contains a Streamlit event-discovery chatbot and an intent classifier trained from clearly disclosed synthetic utterances.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app creates `intent_classifier.joblib` from `eventarena_synthetic_intents.csv` on first run if the model file is absent. To create the model before running, execute `python train_classifier.py`.

## Intents
The classifier has seven labels: `sport`, `concert`, `festival`, `cheap`, `help`, `exit`, and `out_of_scope`. Greeting and name recognition remain rule-based. Requests below the confidence threshold (default 0.30) or classified as `out_of_scope` reach the fallback.

## Synthetic data disclosure
Training utterances are synthetic, hand-written examples for a prototype. They are not real customer conversations. The 0.30 threshold is a demonstration setting carried over from the supplied Part 1 notebook context; that notebook was for Hotel Booking, not EventArena, so this threshold has not been independently validated for this EventArena model. State this limitation in your submission.

## Rubric evidence still to capture
Capture real screenshots after running/deploying:
1. Greeting and name rules.
2. Three classifier-routed intents using natural phrasing.
3. An out-of-scope fallback.
4. Two separate browser sessions with independent conversations.

Do not fabricate evidence. Do not commit passwords, API keys, bot tokens, or other secrets.

## Deploy
On Streamlit Community Cloud, choose this repository, branch `main`, and app path `streamlit/app.py`.
