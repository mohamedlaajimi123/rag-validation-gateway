================================================================================
SECTION 1: PRODUCTION REPOSITORY SETUP & MANIFEST CONFIGURATION
================================================================================

# 1.1 Terminal Directives for Workspace Initialization
mkdir nestjs-rag-gateway
cd nestjs-rag-gateway
git init

# 1.2 Configuration of Environment Exclusions (.gitignore)
cat << 'EOF' > .gitignore
node_modules/
dist/
.env
npm-debug.log*
.DS_Store
EOF

# 1.3 Installation of Declared Codebase Dependencies
npm install @nestjs/config @qdrant/js-client-rest axios form-data

================================================================================
SECTION 2: ACADEMIC-GRADE REPOSITORY DOCUMENTATION (README.md)
================================================================================

# Tiered Validation and Self-Correction Gateway for Document-Grounded RAG Systems

This repository contains the core implementation of a high-performance NestJS gateway service positioned between client user interfaces and a decoupled Python inference execution node. The system implements a deterministic, multi-gate validation framework engineered to eliminate large language model (LLM) hallucinations and enforce strict adherence to bounded document contexts during Retrieval-Augmented Generation (RAG).

## 1. System Architecture and Token Processing Flow

The gateway operates as a synchronous pipeline that handles incoming semantic string queries, orchestrates dense vector database operations, and applies layered validation boundaries before a response payload is cleared for client delivery.

          +------------------------+
          |   Inbound User Query   |
          +-----------+------------+
                      |
                      v
          +------------------------+
          | Ollama Embedding Model |
          |     (all-minilm)       |
          +-----------+------------+
                      |
                      v
          +------------------------+
          | Qdrant Vector Lookup   |
          |  (Dynamic Workspace)   |
          +-----------+------------+
                      |
                      v
          +------------------------+
          |  Lexical Match Filter  | === [Score >= 0.30] ===> [PASSED: Gate 1]
          |       (Gate 1)         |
          +-----------+------------+
                      | (Score < 0.30)
                      v
          +------------------------+
          | Semantic Cosine Align  | === [Score >= 0.40] ===> [PASSED: Gate 2]
          |       (Gate 2)         |
          +-----------+------------+
                      | (Score < 0.40)
                      v
          +------------------------+
          | Cognitive LLM Judge    | === [Verdict = PASSED] => [PASSED: Gate 3]
          |       (Gate 3)         |
          +-----------+------------+
                      | (Verdict = FAILED)
                      v
          +------------------------+
          |   System Intercept /   |
          |   Pipeline Halted      |
          +------------------------+

1. Token Vectorization: Incoming unstructured string queries are transformed into structural embeddings via a local Ollama embedding engine utilizing the all-minilm transformer model.
2. Dynamic Workspace Spatial Indexing: The generated dense vector matrix is mapped against an active partition inside a Qdrant vector database (user_dynamic_workspace collection layout), extracting the top 5 nearest-neighbor context blocks.
3. Node Communication Loop: The compiled text context and user queries are formatted into explicit request interfaces and dispatched to a secondary Python inference node for token generation.

## 2. Core Concept: The Multi-Gate Grounding Engine

The core technical contribution of this gateway is its progressive evaluation pipeline, which evaluates the structural accuracy of model inferences against source documents using three isolated validation layers.

### Gate 1: Token-Frequency Lexical Filter (Stochastic Token Alignment)
The baseline gate runs an optimized string-overlap analysis between the generated answer text and the raw background context retrieved from the database.
* Mechanism: The architecture extracts text tokens, ignores domain-agnostic stop words, and processes token frequencies (unigrams) alongside consecutive text arrays (bigrams).
* Evaluation Metric: A deterministic math score maps unigram overlap at a 60% weight and consecutive bigram layout positioning at a 40% weight.
* Threshold Condition: Matches matching or exceeding the threshold limit of 0.30 clear validation instantly, assigning the payload a status token of VERIFIED_BY_LEXICAL_MATCH.

### Gate 2: High-Dimensional Semantic Cosine Alignment (Vector Proximity Gate)
If conceptual paraphrasing or complex synthesis causes the inference payload to drop below literal keyword metrics, the gateway shifts validation to a dense vector matching layer.
* Mechanism: The generated answer text is vectorized in real-time. Simultaneously, retrieved database text blocks are mapped into individual vector coordinates.
* Evaluation Metric: The gateway processes the exact cosine similarity (the dot product of vector coordinates divided by the product of their Euclidean lengths) across all matching blocks.
* Threshold Condition: If the maximum computed segment alignment score matches or exceeds 0.40, the statement is approved via proximity and logged as VERIFIED_BY_SEMANTIC_SIMILARITY.

### Gate 3: Low-Temperature Cognitive LLM Audit (The Ultimate Arbitrator)
When a response yields borderline semantic vectors (falling below the 0.40 threshold but remaining above the absolute rejection floor of 0.22), it triggers the final cognitive evaluation layer.
* Mechanism: The generated inference string and an abbreviated 1500-character snapshot of the source text are routed to an independent, local LLM instance (llama3) serving as a factual compliance auditor.
* Engineering Parameters: To enforce complete predictability and stop secondary hallucinations inside the security code itself, the auditing LLM is restricted to a temperature configuration of 0.0 and an execution limit of 3 tokens.
* Threshold Condition: The auditor must execute a strict binary verdict: PASSED or FAILED. If the model catches an ungrounded claim or extrapolation missing from the source context, it triggers a system exception, terminates the transaction, and substitutes a secure intercept message to block text contamination.

### Macro-Intent Bypass Layer
To minimize processing latency on highly repetitive, structural baseline inquiries, the gateway runs an early-stage intent normalization loop. When query tokens strictly match designated macro criteria (such as structural network definitions), the system routes an approved baseline payload directly to Gate 1 at an authenticated grounding score of 95%.

## 3. Configuration and Deployment

### Environmental Requirements
Verify the availability of the following endpoints inside your cluster environment:
- NestJS Gateway Service: Port 3001
- Python Inference Compute Worker: Port 5000
- Qdrant Vector Database Server: Port 6333
- Ollama Engine Server: Port 11434

### Direct Execution

To spin up the complete validation pipeline locally, you must execute both the core AI inference node and the API gateway service in separate terminal windows.

#### Phase 1: Initialize the Python Inference Worker Node
Open a terminal at the root directory, navigate to your Python directory, configure a virtual environment, install the dependencies, and boot the execution server:
```bash
# Navigate to the inference service directory
cd python-ai-service

# Initialize and activate an isolated virtual environment
python -m venv venv
source venv/Scripts/activate  # On Linux/macOS use: source venv/bin/activate

# Install required inference and machine learning libraries
pip install -r requirements.txt

# Launch the secure inference server
python server.py

#### Phase 2: Initialize the NestJS Gateway
Open a second terminal window at the root directory, navigate to your gateway directory, resolve the node manifests, and spin up the multi-gate execution pipeline:
```bash
# Navigate to the gateway API directory
cd nestjs-gateway

# Install package dependencies listed in manifest
npm install

# Run the complete automated End-to-End integration suite
npm run test:e2e

# Compile codebase and spin up the gateway instance
npm run start:dev
