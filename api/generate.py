import os

from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def gpt(query: str, context: str):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Du bist ein intelligenter Assistent.
                            Basierend auf dem untenstehenden Kontext, erstelle
                            eine kohärente und informative Antwort 
                           auf die Frage des Nutzers.""",
            },
            {"role": "assistant", "content": f"Kontext: {context}"},
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0].message
