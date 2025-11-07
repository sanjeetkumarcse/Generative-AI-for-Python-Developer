from flask import Flask, jsonify, request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from time import time

app = Flask(__name__)

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    "flask_app_requests_total", "Total number of requests", ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "flask_app_request_latency_seconds", "Request latency (seconds)", ["endpoint"]
)

@app.before_request
def start_timer():
    request.start_time = time()

@app.after_request
def record_metrics(response):
    latency = time() - request.start_time
    REQUEST_LATENCY.labels(request.path).observe(latency)
    REQUEST_COUNT.labels(request.method, request.path, response.status_code).inc()
    return response

@app.route("/")
def index():
    return jsonify({"message": "Hello from Flask with Prometheus!"})

@app.route("/metrics")
def metrics():
    """Expose metrics for Prometheus to scrape"""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    print("🚀 Flask app running at http://localhost:8000")
    app.run(host="0.0.0.0", port=8000)
