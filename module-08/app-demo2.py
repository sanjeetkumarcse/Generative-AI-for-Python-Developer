from flask import Flask, jsonify, request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
    Info,
)
import psutil
from time import time
import random

app = Flask(__name__)

# ---------------------------------
# 🔹 1. Core Web Request Metrics
# ---------------------------------
REQUEST_COUNT = Counter(
    "flask_app_requests_total",
    "Total number of requests",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "flask_app_request_latency_seconds",
    "Request latency (seconds)",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]  # Fine-grained latency buckets
)

# ---------------------------------
# 🔹 2. Error & Success Metrics
# ---------------------------------
ERROR_COUNT = Counter(
    "flask_app_request_error_total",
    "Total number of errors",
    ["endpoint", "error_type"]
)

SUCCESS_COUNT = Counter(
    "flask_app_request_success_total",
    "Total number of successful responses",
    ["endpoint"]
)

# ---------------------------------
# 🔹 3. LLM/AI-specific Metrics (if used)
# ---------------------------------
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens processed by the model",
    ["type"]  # input or output
)

LLM_COST_USD_TOTAL = Counter(
    "llm_cost_usd_total",
    "Total cost incurred by token usage (in USD)"
)

LLM_CACHE_HIT_RATIO = Gauge(
    "llm_cache_hit_ratio",
    "Ratio of cache hits to total LLM requests"
)

# ---------------------------------
# 🔹 4. System & Resource Metrics
# ---------------------------------
CPU_USAGE = Gauge("system_cpu_usage_percent", "Current CPU usage (%)")
MEMORY_USAGE = Gauge("system_memory_usage_percent", "Current memory usage (%)")
UPTIME = Gauge("app_uptime_seconds", "Time since app started (seconds)")
ACTIVE_REQUESTS = Gauge("flask_active_requests", "Number of active in-progress requests")

# ---------------------------------
# 🔹 5. Business / App Metrics
# ---------------------------------
USER_SESSIONS = Counter("app_user_sessions_total", "Number of user sessions started")
FEEDBACK_SCORE = Gauge("app_feedback_score_average", "Average user feedback score (1–5)")

# ---------------------------------
# 🔹 6. App Info Metadata
# ---------------------------------
APP_INFO = Info("flask_app_info", "Application info")
APP_INFO.info({"version": "1.2.3", "framework": "Flask", "env": "dev"})

# ---------------------------------
# Middleware to track requests
# ---------------------------------
app_start_time = time()

@app.before_request
def before_request():
    request.start_time = time()
    ACTIVE_REQUESTS.inc()

@app.after_request
def after_request(response):
    latency = time() - request.start_time
    endpoint = request.path
    REQUEST_LATENCY.labels(endpoint).observe(latency)
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()

    if response.status_code >= 500:
        ERROR_COUNT.labels(endpoint, "server_error").inc()
    elif response.status_code >= 400:
        ERROR_COUNT.labels(endpoint, "client_error").inc()
    else:
        SUCCESS_COUNT.labels(endpoint).inc()

    ACTIVE_REQUESTS.dec()
    return response

# ---------------------------------
# Example routes
# ---------------------------------
@app.route("/")
def index():
    return jsonify({"message": "Hello from Flask with Extended Prometheus!"})

@app.route("/generate")
def generate_text():
    """Simulate LLM text generation"""
    input_tokens = random.randint(100, 500)
    output_tokens = random.randint(200, 1000)
    cost = (input_tokens + output_tokens) / 1000 * 0.005  # $0.005 per 1K tokens

    LLM_TOKENS_TOTAL.labels("input").inc(input_tokens)
    LLM_TOKENS_TOTAL.labels("output").inc(output_tokens)
    LLM_COST_USD_TOTAL.inc(cost)
    LLM_CACHE_HIT_RATIO.set(random.uniform(0.7, 0.95))

    return jsonify({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost
    })

@app.route("/feedback/<int:score>")
def feedback(score):
    """Simulate user feedback"""
    if 1 <= score <= 5:
        FEEDBACK_SCORE.set(score)
        return jsonify({"message": f"Feedback recorded: {score}"})
    return jsonify({"error": "Invalid score"}), 400

@app.route("/metrics")
def metrics():
    """Expose metrics for Prometheus to scrape"""
    # Update system metrics
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    UPTIME.set(time() - app_start_time)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# ---------------------------------
# Error handlers
# ---------------------------------
@app.errorhandler(404)
def not_found(e):
    ERROR_COUNT.labels("/404", "not_found").inc()
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    ERROR_COUNT.labels("/500", "internal_error").inc()
    return jsonify({"error": "Internal Server Error"}), 500

# ---------------------------------
# App startup
# ---------------------------------
if __name__ == "__main__":
    print("🚀 Flask app running at http://localhost:8000")
    app.run(host="0.0.0.0", port=8000)
