# Multi-Gate Grounding Gateway for RAG Systems

This repository contains a NestJS gateway service connected to a Python inference node. The system runs incoming user queries through a three-stage validation pipeline to verify that generated answers are accurately grounded in your source documents, preventing model hallucinations.

---

## 1. System Architecture

The gateway handles incoming text queries, manages vector lookups, and applies layered validation filters before delivering the final response.

```
+------------------------+
|   Inbound User Query   |
+-----------+------------+
            |
            v
+------------------------+
| Ollama Embedding Model |
|      (all-minilm)      |
+-----------+------------+
            |
            v
+------------------------+
|  Qdrant Vector Lookup  |
+-----------+------------+
            |
            v
+------------------------+
|  Keyword Match Filter  | === [Score >= 0.30] ===> [PASSED: Gate 1]
|        (Gate 1)        |
+-----------+------------+
            | (Score < 0.30)
            v
+------------------------+
| Semantic Cosine Match  | === [Score >= 0.40] ===> [PASSED: Gate 2]
|        (Gate 2)        |
+-----------+------------+
            | (Score < 0.40)
            v
+------------------------+
|     LLM Audit Rule     | === [Verdict = PASSED] => [PASSED: Gate 3]
|        (Gate 3)        |
+-----------+------------+
            | (Verdict = FAILED)
            v
+------------------------+
|    Pipeline Halted /   |
|    Secure Intercept    |
+------------------------+
```

1. **Vectorization:** User queries are converted into text embeddings locally using Ollama (`all-minilm`).
2. **Vector Search:** The gateway queries a Qdrant collection to retrieve the top 5 relevant document context blocks.
3. **Service Communication:** The retrieved context and original query are forwarded to the Python inference node to generate the initial answer.

---

## 2. The Three-Stage Validation Pipeline

The core mechanism of this project is a progressive validation workflow that evaluates the generated answer against the source context using three layers.

### Gate 1: Keyword Match Filter

This layer runs a literal text-overlap comparison between the generated answer and the source document text.

- **Mechanism:** Cleans the text, removes common stop words, and checks for overlapping unigrams and bigrams.
- **Metric:** A weighted score combining 60% unigram overlap and 40% bigram proximity.
- **Threshold:** A score of **0.30 or higher** passes immediately, marking the payload as verified.

### Gate 2: Semantic Cosine Match

If paraphrasing or conceptual language causes the keyword score to fall below 0.30, the pipeline shifts to vector similarity.

- **Mechanism:** Generates text vectors for both the generated answer and the retrieved context blocks.
- **Metric:** Cosine similarity between the coordinate vectors.
- **Threshold:** A score of **0.40 or higher** approves the answer via semantic proximity.

### Gate 3: LLM Audit Rule Check

If the response falls into a borderline similarity zone (below 0.40 but above the absolute rejection floor of 0.22), a final factual check is triggered.

- **Mechanism:** The generated response and a 1,500-character snapshot of the source document are sent to a local `llama3` instance acting as a strict quality auditor.
- **Parameters:** The auditing model runs at `temperature: 0.0` with a maximum output of 3 tokens for fully deterministic output.
- **Threshold:** The model returns a strict binary verdict — `PASSED` or `FAILED`. If any claim is unsupported by the source context, the transaction is intercepted, blocked, and replaced with a secure fallback message.

---

## 3. Evaluation and Experimental Setup

### Dataset Grounding

The pipeline's accuracy parameters were verified against a test suite of 100 document-grounded query-context pairs, including baseline factual inquiries, conceptual paraphrases, and deliberate adversarial inputs designed to trigger hallucinations.

### Threshold Calibration

Boundary scores were determined empirically to minimize false-negative rejections while catching out-of-context claims:

- **Gate 1 (0.30 — Lexical):** Calibrated to clear direct quotes and high-keyword-overlap responses instantly, bypassing the cost of vector generation.
- **Gate 2 (0.40 — Semantic):** Calibrated using cosine similarity bounds to catch accurate conceptual summaries that use alternate vocabulary.
- **Gate 3 (0.22 — Absolute Rejection Floor):** Context pairs scoring below 0.22 are automatically classified as fully irrelevant, bypassing the LLM auditor to reduce token processing latency.

### Performance Profile

In baseline testing against ungrounded Llama3 responses, the three-stage pipeline successfully intercepted out-of-bounds hallucinations while maintaining minimal processing latency overhead.

---

## 4. Configuration and Deployment

### Environmental Requirements

Ensure the following local endpoints are accessible in your environment:

| Service | Port |
|---|---|
| NestJS Gateway | `3001` |
| Python Inference Worker | `5000` |
| Qdrant Vector Database | `6333` |
| Ollama Engine Server | `11434` |

### Direct Execution

To run the system locally, use two separate terminal sessions.

#### Phase 1 — Initialize the Python Inference Worker

Open your first terminal at the workspace root:

```bash
cd python-ai-service

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the inference server
python server.py
```

#### Phase 2 — Initialize the NestJS Gateway

Open a second terminal at the workspace root:

```bash
cd nestjs-gateway

# Install dependencies
npm install

# Run the end-to-end integration test suite
npm run test:e2e

# Start the gateway service
npm run start:dev
```
