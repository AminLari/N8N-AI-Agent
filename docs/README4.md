```markdown
# Fraud Detection with LLM-Augmented Anomaly Detection

## Project Overview
A **proof-of-concept (PoC)** for integrating **Large Language Models (LLMs)** with traditional **anomaly detection** in transaction monitoring at Scotiabank. The system combines **unsupervised ML (Autoencoders)** for baseline anomaly detection with **RAG-based LLM reasoning** to explain fraudulent transactions dynamically.

---

## Key Features
1. **Hybrid Anomaly Detection**
   - Uses **Autoencoder (TensorFlow/Keras)** to model normal transaction patterns.
   - Detects deviations via reconstruction error thresholds.

2. **LLM-Augmented Explanations**
   - Leverages **LangChain + Mistral API** to generate human-readable fraud flags.
   - Retrieves context from transaction history via **RAG** before flagging.

3. **Scalability**
   - Deployed as a **microservice** (FastAPI) with batch processing for high-volume transactions.

---

## Tech Stack
- **Core ML**: TensorFlow (Autoencoder), PyTorch (Baseline)
- **LLM Integration**: LangChain, Mistral API, RAG (ChromaDB)
- **Deployment**: Docker, Kubernetes (for scalability)
- **Monitoring**: Prometheus + Grafana

---

## Demo Workflow
1. **Input**: Raw transaction data (amount, timestamp, merchant category).
2. **Step 1**: Autoencoder reconstructs transactions; high reconstruction error → flagged as anomaly.
3. **Step 2**: LLM (Mistral) analyzes flagged transactions via RAG to:
   - Cross-reference with historical patterns.
   - Flag suspicious patterns (e.g., sudden high-value transactions in a new merchant category).
4. **Output**: Alerts with **human-readable explanations** (e.g., *"Transaction $1200 at Acme Corp may be fraudulent due to 3x higher spending than usual for this merchant category."*).

---

## Sample Code Snippet (Autoencoder + LLM Integration)
```python
# Autoencoder for baseline anomaly detection
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

model = tf.keras.Sequential([
    Input(shape=(10,)),  # 10 features per transaction
    Dense(64, activation='relu'),
    Dense(10)
])

model.compile(optimizer='adam', loss='mse')

# LLM explanation generator (simplified)
def explain_fraud(transaction, llm_model):
    context = f"Transaction: {transaction}. Is this fraudulent? Provide reasoning."
    response = llm_model.generate(context)
    return response.strip()
```

---

## Impact Metrics
| Metric               | Target Outcome                     |
|----------------------|------------------------------------|
| False Positive Rate  | <5% (adjustable via threshold)     |
| Fraud Detection Rate | 90%+ (vs. baseline 70%)           |
| User Satisfaction    | 85%+ (via post-alert surveys)      |

---

## How to Run Locally
1. Clone repo: `git clone https://github.com/your-repo/scotiabank-fraud-llm-demo.git`
2. Install dependencies: `pip install tensorflow langchain chromadb fastapi`
3. Start services:
   ```bash
   docker-compose up --build
   ```
4. Test API:
   ```bash
   curl -X POST http://localhost:8000/flag \
     -H "Content-Type: application/json" \
     -d '{"transaction": {"amount": 1200, "merchant": "Acme Corp"}}'
   ```

---
## Future Enhancements
- **Real-time processing**: Stream transactions via Kafka.
- **Active learning**: Use LLM to flag ambiguous cases for human review.
- **Multi-modal analysis**: Integrate transaction images (e.g., receipts) via OCR + LLM.

---
## Files
- `data/transactions.csv` – Sample transaction dataset.
- `models/autoencoder.h5` – Pre-trained Autoencoder model.
- `llm_examples/` – Example prompts for Mistral API.
- `docker-compose.yml` – Deployment setup.
```