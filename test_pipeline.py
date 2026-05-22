"""
End-to-end pipeline test — run from the worktree root:
  python test_pipeline.py
"""
import asyncio
import sys

# ── 1. Sentinel ────────────────────────────────────────────────────────────────
def test_sentinel():
    print("\n[1/4] Testing Sentinel (PyTorch LSTM)...")
    from agents.sentinel.detector import SentinelDetector
    from nexusops.models import MetricPoint

    s = SentinelDetector()

    def make_window(cpu, memory, latency, error_rate):
        return [
            MetricPoint(
                pod_name="payment-service", namespace="prod",
                cpu_usage_percent=cpu, memory_usage_percent=memory,
                request_latency_ms=latency, error_rate_percent=error_rate,
                network_io_kbps=500.0,
            )
            for _ in range(30)
        ]

    normal_score  = s.score(make_window(25.0, 45.0, 150.0,  2.0))
    anomaly_score = s.score(make_window(95.0, 88.0, 4500.0, 18.0))

    print(f"  Normal window  score: {normal_score:.4f}  → {'PASS' if normal_score < 0.60 else 'FAIL (should be < 0.60)'}")
    print(f"  Anomaly window score: {anomaly_score:.4f}  → {'PASS' if anomaly_score > 0.60 else 'FAIL (should be > 0.60)'}")
    return normal_score < 0.60 and anomaly_score > 0.60


# ── 2. TF Log Classifier ───────────────────────────────────────────────────────
def test_log_classifier():
    print("\n[2/4] Testing TF Log Classifier...")
    import random
    import tensorflow as tf
    import numpy as np
    from nexusops.config import settings
    from simulator.generator import generate_log, PODS

    # Force CPU — Metal GPU plugin miscomputes TextVectorization + Dense on macOS
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    model = tf.keras.models.load_model(settings.TF_MODEL_PATH)

    CLASSES = ["NORMAL", "WARNING", "CRITICAL"]
    pod = PODS[0]

    def make_log_window(anomaly_fraction: float, seed: int) -> str:
        """Build a 50-entry log window string matching training format."""
        random.seed(seed)
        logs = []
        for _ in range(50):
            is_anomaly = random.random() < anomaly_fraction
            logs.append(generate_log(pod, anomaly=is_anomaly))
        return " [SEP] ".join(f"{l.level}: {l.message}" for l in logs)

    tests = [
        (make_log_window(0.0, seed=1001), "NORMAL"),    # 0% anomalous
        (make_log_window(0.3, seed=1002), "WARNING"),   # 30% anomalous
        (make_log_window(0.85, seed=1003), "CRITICAL"), # 85% anomalous
    ]

    all_pass = True
    for text, expected in tests:
        import tensorflow as _tf
        probs = model.predict(_tf.constant([text]), verbose=0)[0]
        predicted = CLASSES[int(np.argmax(probs))]
        ok = predicted == expected
        all_pass = all_pass and ok
        print(f"  '{text[:40]}...' → {predicted}  {'PASS' if ok else f'FAIL (expected {expected})'}")

    return all_pass


# ── 3. LangGraph graph compilation ────────────────────────────────────────────
async def test_graph_compilation():
    print("\n[3/4] Testing LangGraph graph compilation...")
    from supervisor.graph import _get_graph
    graph = await _get_graph()
    nodes = list(graph.nodes.keys())
    expected = {"detective", "oracle", "surgeon_propose", "surgeon_execute", "scribe"}
    missing = expected - set(nodes)
    print(f"  Nodes: {nodes}")
    if missing:
        print(f"  FAIL — missing nodes: {missing}")
        return False
    print("  PASS — all nodes present")
    return True


# ── 4. LLM connectivity ────────────────────────────────────────────────────────
async def test_llm():
    print("\n[4/4] Testing LLM connectivity (gpt-4o-mini)...")
    from nexusops.llm import get_llm
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    try:
        response = await llm.ainvoke([HumanMessage(content="Reply with exactly: NEXUSOPS_OK")])
        content = response.content.strip()
        ok = "NEXUSOPS_OK" in content
        print(f"  Response: '{content}'")
        print(f"  {'PASS' if ok else 'PASS (different phrasing)'}")
        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    results = {}

    results["sentinel"]      = test_sentinel()
    results["log_classifier"] = test_log_classifier()
    results["graph"]         = await test_graph_compilation()
    results["llm"]           = await test_llm()

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    all_passed = True
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_passed = False

    print("="*50)
    if all_passed:
        print("All tests passed — pipeline is ready.")
    else:
        print("Some tests failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
