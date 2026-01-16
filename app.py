# A small teaching assistant chatbot app
# Powered by Open AI
# As much to help students as demonstrate a product that can be built
# With AI assisted programming.

import os
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dashbootc
from openai import OpenAI
import numpy as np
from dash import dcc

# --------------------
# Environment & client
# --------------------
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# --------------------
# Mood-based helpers
# --------------------
seed = {
    "thinking": [
        "Let me think this through.",
        "I'm considering the options.",
        "Hmm, let's reason step by step."
    ],
    "happy": [
        "Nice, that worked!",
        "Great job, this is going well.",
        "I’m happy with this result."
    ],
    "cheering": [
        "Woohoo! Amazing news!",
        "Fantastic! High five!",
        "Yes! We nailed it!"
    ],
    "frustrated": [
        "This is annoying and not working.",
        "I'm stuck and frustrated.",
        "Ugh, this keeps failing."
    ]
}


def embed_text(texts):
    if isinstance(texts, str):
        texts = [texts]
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return np.array([d.embedding for d in resp.data])


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def build_centroids(seed_dict):
    centroids = {}
    for mood, phrases in seed_dict.items():
        embs = embed_text(phrases)
        centroids[mood] = embs.mean(axis=0)
    return centroids


CENTROIDS = None

def detect_mood(message):
    global CENTROIDS
    if CENTROIDS is None:
        CENTROIDS = build_centroids(seed)

    e = embed_text(message)[0]
    sims = {m: cosine(e, c) for m, c in CENTROIDS.items()}
    return max(sims, key=sims.get)

MOOD_IMAGES = {
    "thinking": "/assets/thinking.png",
    "happy": "/assets/neutral.png",
    "cheering": "/assets/happy.png",
    "frustrated": "/assets/frustrated.png",
}

def mood_img(mood):
    return MOOD_IMAGES.get(mood, "/assets/neutral.png")


# --------------------
# System prompt
# --------------------
SYSTEM_PROMPT = """
You are a teaching assistant for the following courses: intro to copilot, intro to AI (ChatGPT, Gemini, Perplexity), Business Analytics, Data Analysis with AI (ChatGPT and Looker Studio), Intro to Power BI,
business analytics in Power Bi, Intro to Python Programming, and Generative AI programming with Python and Claude.

Give answers in the following way:
- If a user asks about working directory, installation of python or other software setup, do not ask follow-up questions; resolve directly and gently.
- For coding, statistics, and busines related questions, prompt users to think and ask clarifying questions.
- If users struggle or seem frustrated, give partial answers first, then full answers upon follow up.
- Keep explanations simple, but use math and formulas if prompted.
- Cite sources when possible.
"""

# --------------------
# Dash app
# --------------------
app = dash.Dash(__name__, external_stylesheets=[dashbootc.themes.BOOTSTRAP])

app.layout = dashbootc.Container([
    # Header / logo
    dashbootc.Row([
        dashbootc.Col(html.Img(src="/assets/not_unsw_logo.png", style={"width": "120px", "float": "right"}))
    ]),

    html.H2("Teaching Assistant Bot"),

    dashbootc.Row([
        dashbootc.Col(
            dashbootc.Card([
                dashbootc.CardBody([
                    html.H4("About this app"),
                    html.P("This teaching assistant bot is designed to help with introductory courses in the Artificial Intelligence Program."),
                    html.P("It guides you by asking clarifying questions, providing simple explanations, and referring to additional sources when possible. This bot was generated with python code, co-written by AI, and is the kind of thing this course will teach you how to make!")
                ])
            ]),
            width=3
        ),

        dashbootc.Col([
            dcc.Loading(
                id="loading-chat",
                type="default",
                children=html.Div(
                    id="chat-window",
                    style={
                        "border": "1px solid #ddd",
                        "padding": "10px",
                        "height": "400px",
                        "overflowY": "auto",
                        "marginBottom": "10px"
                    }
                )
            ),
            dashbootc.Input(
                id="user-input",
                placeholder="Ask a question...",
                type="text"
            ),
            dashbootc.Button(
                "Send",
                id="send-btn",
                color="primary",
                className="mt-2"
            )
        ], width=9)


        
        
    ]),

    html.Hr(),
    html.Footer(
        "Disclaimer: This bot is for educational purposes only and should not be relied upon for professional or legal advice. This bot uses OpenAI models and is not an official product of UNSW. AI can make mistakes - always check answers and advice yourself.",
        style={"textAlign": "center", "fontSize": "0.8em", "color": "gray"}
    ),

    dcc.Store(id="chat-history", data=[])
], fluid=True)


# --------------------
# Callbacks
# --------------------
@app.callback(
    Output("chat-window", "children"),
    Output("chat-history", "data"),
    Output("user-input", "value"),
    Input("send-btn", "n_clicks"),
    State("user-input", "value"),
    State("chat-history", "data"),
    prevent_initial_call=True
)


def chat(n_clicks, user_text, history):
    if not user_text:
        return dash.no_update, dash.no_update

    mood = detect_mood(user_text)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )

    assistant_text = resp.choices[0].message.content

    history.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text, "mood": mood}
    ])

    rendered = []

    for h in history:
        role = "You" if h["role"] == "user" else "TA Bot"
    
        if h["role"] == "assistant":
            rendered.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=mood_img(h.get("mood")),
                                    style={
                                        "height": "40px",
                                        "marginRight": "10px"
                                    }
                                ),
                                html.B(f"{role}: ")
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center"
                            }
                        ),
                        dcc.Markdown(h["content"])
                    ],
                    style={"marginBottom": "15px"}
                )
            )
        else:
            rendered.append(
                html.Div(
                    [
                        html.B(f"{role}: "),
                        dcc.Markdown(h["content"])
                    ],
                    style={"marginBottom": "15px"}
                )
            )
    
    return rendered, history, ""

# For a local run
#if __name__ == "__main__":
#    app.run(debug=True, host="127.0.0.1", port=8050)

# To run on dash
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050))
    )
