from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from langchain_openai import AzureChatOpenAI
import os, time

# Load env variables
api_key="3KZOVIIWVSSadEQyVLMr722pVoXVOl6BvatHtqrAsnmSV06FL1SCJQQJ99BIACHYHv6XJ3w3AAAAACOGGKR1"
model_name="gpt-4o"
api_version="2024-10-21"
azure_endpoint="https://ai-new-agent-resource.openai.azure.com/"


app = Flask(__name__)

# -------------------------------
# Prometheus Metrics
# -------------------------------
REQUEST_COUNT = Counter(
    "flask_app_requests_total",
    "Total number of requests",
    ["endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "flask_app_request_latency_seconds",
    "Request latency (seconds)",
    ["endpoint"]
)

ERROR_COUNT = Counter(
    "flask_app_request_error_total",
    "Total number of request errors",
    ["endpoint", "error_type"]
)

# -------------------------------
# Azure OpenAI (LangChain) Setup
# -------------------------------
llm = AzureChatOpenAI(
    azure_deployment=model_name,
    openai_api_key=api_key,
    openai_api_version=api_version,
    azure_endpoint=azure_endpoint,
    temperature=0.7,
)

# -------------------------------
# Flask Endpoints
# -------------------------------
@app.route("/ask", methods=["POST"])
def ask_model():
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt", "Hello GPT-4o!")

        # Call the model via LangChain
        response = llm.invoke(prompt)

        latency = time.time() - start_time
        REQUEST_LATENCY.labels("/ask").observe(latency)
        REQUEST_COUNT.labels("/ask", 200).inc()

        return jsonify({"response": response.content, "latency": latency})

    except Exception as e:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels("/ask").observe(latency)
        ERROR_COUNT.labels("/ask", type(e).__name__).inc()
        REQUEST_COUNT.labels("/ask", 500).inc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/")
def home():
    return jsonify({"message": "Azure GPT-4o Flask App is running!"})


# -------------------------------
# Run the Flask app
# -------------------------------
if __name__ == "__main__":
    print("🚀 Running Flask app at http://localhost:8000")
    app.run(host="0.0.0.0", port=8000)
