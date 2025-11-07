from flask import Flask, render_template, request, jsonify
from langchain_openai import AzureChatOpenAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Azure OpenAI model via LangChain
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7
)

 

# Setup conversation chain with memory
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Run conversation
    response = conversation.run(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
