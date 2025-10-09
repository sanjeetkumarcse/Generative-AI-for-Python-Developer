

from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
import os

app = Flask(__name__)

# ====== CONFIGURATION ======
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_KEY = ""
DEPLOYMENT_NAME = "gpt-4o"  # e.g., "gpt-4o-mini"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-05-01-preview"
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"reply": "Error: Empty message", "usage": {}}), 400

        chat_history = request.json.get("history", [])
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        for msg in chat_history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=300
        )

        reply = response.choices[0].message.content

        # Convert usage object to dict
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0)
            }

        return jsonify({"reply": reply, "usage": usage})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}", "usage": {}}), 500

if __name__ == "__main__":
    app.run(debug=True)
