import requests
import json
import time

# Gateway Endpoint
GATEWAY_URL = "http://localhost:3001/api/rag/ask"

# Test Suites carefully calibrated to evaluate each operational layer
TEST_CASES = [
    {
        "id": "TC-001",
        "type": "Direct Extraction (Gate 1 Target)",
        "prompt": "What is the primary problem with peer-to-peer electronic cash described in the introduction?"
    },
    {
        "id": "TC-002",
        "type": "Conceptual Paraphrase (Gate 2 Target)",
        "prompt": "Explain how this digital ledger architecture completely blocks users from duplicating transactions."
    },
    {
        "id": "TC-003",
        "type": "Fuzzy Synthesis / Gray Zone (Gate 3 Target)",
        "prompt": "Is the privacy model in this peer-to-peer network completely anonymous or pseudonymous? Synthesize an overview."
    },
    {
        "id": "TC-004",
        "type": "Out-of-Bounds Hallucination (Rejection Target)",
        "prompt": "What are the step-by-step instructions for changing a flat tire on a modern passenger car?"
    }
]

print("=" * 80)
print("🚀 ENTERPRISE SELF-CORRECTING RAG GATEWAY - AUTOMATED BENCHMARK RUNNER")
print("=" * 80)

results_matrix = []

for case in TEST_CASES:
    print(f"\n▶️ Executing [{case['id']}] | Category: {case['type']}")
    print(f"   Prompt: \"{case['prompt']}\"")
    
    start_time = time.time()
    try:
        response = requests.post(
            GATEWAY_URL, 
            json={"question": case["prompt"]},
            headers={"Content-Type": "application/json"}
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            score_pct = f"{(data.get('groundingScore', 0) * 100):.0f}%"
            
            results_matrix.append({
                "id": case["id"],
                "status": data.get("status"),
                "score": score_pct,
                "latency": f"{latency:.0f}ms"
            })
            print(f"   ✅ Success | Status Badge: {data.get('status')} | Score: {score_pct} | Latency: {latency:.0f}ms")
        else:
            print(f"   ❌ Gateway Error: Received HTTP Status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection Failed: Check if your NestJS server is running on Port 3000. Error: {e}")
    
    time.sleep(1) # Polite execution spacing

print("\n" + "=" * 80)
print("📊 FINAL SYSTEM PERFORMANCE EVALUATION MATRIX")
print("=" * 80)
print(f"{'ID':<8} | {'DETERMINISTIC GATE GATEWAY STATUS':<35} | {'SCORE':<6} | {'LATENCY':<8}")
print("-" * 80)
for res in results_matrix:
    print(f"{res['id']:<8} | {res['status']:<35} | {res['score']:<6} | {res['latency']:<8}")
print("=" * 80)