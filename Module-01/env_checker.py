import importlib
import os
import sys
from datetime import datetime

# ===== Recommended Python Version =====
MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 11)

# ===== Required Packages & Versions =====
REQUIRED_PACKAGES = {
    # Core Python
    "numpy": "1.24.0",
    "pandas": "2.0.0",
    "matplotlib": "3.7.0",
    "scikit-learn": "1.7.2",
    "jupyterlab": "4.0.0",

    # Deep Learning
    "torch": "2.0.0",
    "torchvision": "0.15.0",
    "torchaudio": "2.8.0",
    "tensorflow": "2.13.0",

    # Hugging Face
    "transformers": "4.34.0",
    "datasets": "2.15.0",
    "accelerate": "0.23.0",
    "evaluate": "0.4.0",

    # Generative AI APIs
    "openai": "1.3.0",
    "python-dotenv": "1.1.1",

    # Image Generation
    "diffusers": "0.21.0",
    "safetensors": "0.4.0",
    "pillow": "10.0.0",

    # RAG & LangChain
    "langchain": "0.0.330",
    "llama-index": "0.9.15",
    "chromadb": "0.4.13",
    "pinecone-client": "2.2.4",
    "sentence-transformers": "2.2.2",

    # Web & App Development
    "flask": "2.3.0",
    "streamlit": "1.27.0",
    "fastapi": "0.103.0",
    "uvicorn": "0.23.0",

    # Agentic AI
    "pyautogen": "0.2.0",
    "crewai": "0.1.10",
    "langgraph": "0.0.20",

    # Monitoring
    "prometheus-client": "0.17.1",

    # Utility
    "requests": "2.31.0",
    "multipart": "0.0.6",
}

# ===== Tracking Results =====
results = {"ok": 0, "outdated": 0, "missing": 0, "error": 0, "unknown": 0}
to_install = []
log_lines = []


def log(msg):
    """Log message to both console and file buffer."""
    print(msg)
    log_lines.append(msg)


def check_package(name, min_version):
    """Check if package is installed and meets min version."""
    try:
        module_name = name.replace("-", "_")
        pkg = importlib.import_module(module_name)
        version = getattr(pkg, "__version__", "unknown")

        if version != "unknown":
            from packaging import version as v
            if v.parse(version) >= v.parse(min_version):
                log(f"✅ {name} OK (v{version})")
                results["ok"] += 1
            else:
                log(f"⚠️ {name} outdated (v{version}, need ≥ {min_version})")
                results["outdated"] += 1
                to_install.append(f"{name}=={min_version}")
        else:
            log(f"ℹ️ {name} installed but version unknown")
            results["unknown"] += 1
    except ImportError:
        log(f"❌ {name} NOT installed")
        results["missing"] += 1
        to_install.append(f"{name}=={min_version}")
    except Exception as e:
        log(f"❌ {name} error while loading → {e}")
        results["error"] += 1
        to_install.append(f"{name}=={min_version}")


def check_api_keys():
    """Check if API keys are set in environment variables."""
    openai_key = os.getenv("OPENAI_API_KEY")
    hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    log("\n🔑 API Key Check:")
    log(f"OpenAI Key: {'✅ Set' if openai_key else '❌ Missing'}")
    log(f"Hugging Face Key: {'✅ Set' if hf_key else '❌ Missing'}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n\n============================")
    log(f"🕒 Environment Check @ {timestamp}")
    log("============================\n")

    # ===== Python Version Check =====
    py_ver = sys.version_info
    log(f"🔍 Python version: {sys.version} ")
    if py_ver < MIN_PYTHON or py_ver > MAX_PYTHON:
        log(f"⚠️ WARNING: Recommended Python version is {MIN_PYTHON[0]}.{MIN_PYTHON[1]} - {MAX_PYTHON[0]}.{MAX_PYTHON[1]}")

    log("=== Checking Required Packages ===")
    for pkg, ver in REQUIRED_PACKAGES.items():
        check_package(pkg, ver)

    check_api_keys()

    # ===== Summary =====
    log("\n📊 Summary Report:")
    log(f"✅ OK: {results['ok']}")
    log(f"⚠️ Outdated: {results['outdated']}")
    log(f"ℹ️ Unknown version: {results['unknown']}")
    log(f"❌ Missing: {results['missing']}")
    log(f"❌ Error loading: {results['error']}")

    # ===== Auto Generate Install Command =====
    if to_install:
        log("\n💡 To fix issues, run:")
        log("pip install " + " ".join(to_install))
    else:
        log("\n🎉 All required packages are correctly installed.")

    # ===== Append Report =====
    with open("env_report.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    log("\n📝 Report appended to env_report.txt")
