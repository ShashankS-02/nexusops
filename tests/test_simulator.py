"""
Tests for the Metrics Simulator
"""

from datetime import datetime

import numpy as np

from nexusops.models import LogEntry, MetricPoint
from simulator.generator import (
    PODS,
    generate_cascading_failure,
    generate_cpu_spike,
    generate_log,
    generate_memory_leak,
    generate_normal_metric,
)


class TestNormalMetricGeneration:
    """Verify that normal metrics stay within expected bounds."""

    def test_cpu_within_bounds(self):
        """CPU should stay within 0–100%."""
        pod = PODS[0]
        for _ in range(100):
            m = generate_normal_metric(pod)
            assert 0.0 <= m.cpu_usage_percent <= 100.0

    def test_memory_within_bounds(self):
        pod = PODS[0]
        for _ in range(100):
            m = generate_normal_metric(pod)
            assert 0.0 <= m.memory_usage_percent <= 100.0

    def test_latency_non_negative(self):
        pod = PODS[0]
        for _ in range(100):
            m = generate_normal_metric(pod)
            assert m.request_latency_ms >= 0.0

    def test_error_rate_within_bounds(self):
        pod = PODS[0]
        for _ in range(100):
            m = generate_normal_metric(pod)
            assert 0.0 <= m.error_rate_percent <= 100.0

    def test_pod_name_preserved(self):
        pod = PODS[1]
        m = generate_normal_metric(pod)
        assert m.pod_name == pod["name"]
        assert m.namespace == pod["namespace"]

    def test_returns_metric_point_instance(self):
        m = generate_normal_metric(PODS[0])
        assert isinstance(m, MetricPoint)

    def test_timestamp_is_set(self):
        m = generate_normal_metric(PODS[0])
        assert isinstance(m.timestamp, datetime)


class TestCpuSpikeScenario:
    """CPU spike should produce elevated CPU readings."""

    def test_cpu_elevated_at_full_intensity(self):
        """At intensity=1.0, CPU should typically be above 75%."""
        pod = PODS[0]
        cpu_values = [generate_cpu_spike(pod, intensity=1.0).cpu_usage_percent for _ in range(50)]
        avg_cpu = np.mean(cpu_values)
        assert avg_cpu > 75.0, f"Expected CPU > 75% on spike, got {avg_cpu:.1f}%"

    def test_cpu_higher_than_normal(self):
        """Spike CPU should be higher than normal CPU on average."""
        pod = PODS[0]
        normal_cpu = np.mean([generate_normal_metric(pod).cpu_usage_percent for _ in range(50)])
        spike_cpu = np.mean(
            [generate_cpu_spike(pod, intensity=0.8).cpu_usage_percent for _ in range(50)]
        )
        assert spike_cpu > normal_cpu

    def test_latency_increases_with_cpu_spike(self):
        """CPU spike should also increase latency."""
        pod = PODS[0]
        normal_lat = np.mean([generate_normal_metric(pod).request_latency_ms for _ in range(50)])
        spike_lat = np.mean(
            [generate_cpu_spike(pod, intensity=1.0).request_latency_ms for _ in range(50)]
        )
        assert spike_lat > normal_lat


class TestMemoryLeakScenario:
    """Memory leak should show increasing memory over time."""

    def test_memory_increases_with_elapsed_time(self):
        """Memory at t=300s should be higher than at t=0."""
        pod = PODS[0]
        early = np.mean(
            [generate_memory_leak(pod, elapsed_seconds=0).memory_usage_percent for _ in range(50)]
        )
        late = np.mean(
            [generate_memory_leak(pod, elapsed_seconds=300).memory_usage_percent for _ in range(50)]
        )
        assert late > early, f"Expected late memory ({late:.1f}%) > early ({early:.1f}%)"

    def test_memory_grows_monotonically(self):
        """Average memory should trend upward over increasing elapsed time."""
        pod = PODS[0]
        times = [0, 60, 120, 240, 480]
        avg_memories = []
        for t in times:
            avg = np.mean([generate_memory_leak(pod, t).memory_usage_percent for _ in range(30)])
            avg_memories.append(avg)

        for i in range(1, len(avg_memories)):
            assert avg_memories[i] >= avg_memories[i - 1], (
                f"Memory not monotonically increasing: {avg_memories}"
            )


class TestCascadingFailureScenario:
    """Cascading failure should spike both latency and error rate."""

    def test_full_failure_has_high_error_rate(self):
        pod = PODS[0]
        errors = [
            generate_cascading_failure(pod, failure_progress=0.95).error_rate_percent
            for _ in range(50)
        ]
        assert np.mean(errors) > 20.0, "Full failure should have >20% error rate"

    def test_full_failure_has_high_latency(self):
        pod = PODS[0]
        latencies = [
            generate_cascading_failure(pod, failure_progress=0.95).request_latency_ms
            for _ in range(50)
        ]
        assert np.mean(latencies) > 2000.0

    def test_early_failure_lower_than_late(self):
        pod = PODS[0]
        early_errors = np.mean(
            [generate_cascading_failure(pod, 0.1).error_rate_percent for _ in range(50)]
        )
        late_errors = np.mean(
            [generate_cascading_failure(pod, 0.9).error_rate_percent for _ in range(50)]
        )
        assert late_errors > early_errors


class TestLogGeneration:
    """Verify log generation for normal and anomalous cases."""

    def test_normal_log_is_info(self):
        """Normal logs should predominantly be INFO level."""
        pod = PODS[0]
        logs = [generate_log(pod, anomaly=False) for _ in range(100)]
        info_count = sum(1 for log in logs if log.level == "INFO")
        assert info_count > 90, f"Expected >90 INFO logs, got {info_count}"

    def test_anomaly_log_has_errors(self):
        """Anomaly logs should be ERROR or CRITICAL level."""
        pod = PODS[0]
        logs = [generate_log(pod, anomaly=True) for _ in range(100)]
        error_count = sum(1 for log in logs if log.level in ("ERROR", "CRITICAL", "WARN"))
        assert error_count == 100, f"All anomaly logs should be error-level, got {error_count}"

    def test_log_returns_log_entry(self):
        log = generate_log(PODS[0])
        assert isinstance(log, LogEntry)

    def test_log_has_trace_id(self):
        log = generate_log(PODS[0])
        assert log.trace_id is not None
        assert len(log.trace_id) > 0
