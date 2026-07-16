# Monitor System Resources

When running latency benchmarks — especially load tests with high concurrency — it's important to know whether the client machine itself is a bottleneck. High CPU usage, memory pressure, or network saturation on the client side can skew latency measurements without any visible errors.

LLMeter's [`SystemMetricsMonitor`](../../reference/callbacks/system_metrics/#llmeter.callbacks.system_metrics.SystemMetricsMonitor) callback tracks CPU, memory, and network I/O during benchmark runs, providing both real-time visibility through the progress display and persisted statistics in the result.

!!! note "Optional dependency"
    System metrics monitoring requires `psutil`. Install with:
    ```bash
    pip install 'llmeter[system-metrics]'
    ```

## Quick start

```python
from llmeter.callbacks.system_metrics import SystemMetricsMonitor
from llmeter.runner import Runner

monitor = SystemMetricsMonitor(sample_interval=0.5, per_process=True)
runner = Runner(endpoint=endpoint, callbacks=[monitor], output_path="./results")
result = await runner.run(payload=payload, clients=10, n_requests=50)

# Aggregated stats are available on the result
print(f"CPU avg: {result.stats['system_cpu_percent-average']:.1f}%")
print(f"Memory peak: {result.stats['system_memory_rss_mb-max']:.1f} MB")
print(f"Network received: {result.stats['system_net_bytes_recv_total']} bytes")
```

## How it works

The monitor spawns a lightweight background thread that periodically samples system metrics while the benchmark runs:

1. **`before_run`** — resets state, captures the starting network counters, and starts the sampling thread.
2. **During the run** — the thread collects CPU, memory, and network samples at the configured `sample_interval`.
3. **`after_run`** — stops the thread and computes aggregated statistics (average, p50, p90, p99, max) that are contributed to `result.stats`.

Because sampling happens on a separate thread, the overhead on benchmark throughput is negligible.

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `sample_interval` | `1.0` | Seconds between samples. Lower values yield more granular data. |
| `per_process` | `True` | If `True`, CPU and memory are scoped to the current Python process. Set to `False` for system-wide monitoring. |

!!! note "Network I/O is always system-wide"
    Regardless of the `per_process` setting, network I/O metrics are always system-wide because `psutil` does not support per-process network counters on most platforms.

## Live display integration

When attached to a Runner, the monitor surfaces real-time metrics in the progress bar during the run — showing current CPU %, RSS memory, and network receive rate. This happens automatically via the `live_stats()` hook; no extra configuration is needed.

## Reuse across runs

A single `SystemMetricsMonitor` instance can be reused across multiple `runner.run()` calls. The monitor resets its internal state in `before_run`, so each run gets independent samples and statistics.

## Detecting client-side bottlenecks

After a run, check these indicators:

- **`system_cpu_percent-p90` near 100%** — your client process is CPU-bound and may not be able to drive requests fast enough.
- **`system_memory_rss_mb-max` growing significantly** — memory pressure could trigger GC pauses or swapping.
- **Network rates plateauing while latency increases** — possible network saturation.

If you observe any of these, consider running LLMeter on a more powerful instance or reducing concurrency.

## Accessing raw samples

For custom analysis (e.g., time-series plots correlating CPU with latency), you can access the raw collected samples:

```python
for sample in monitor.samples:
    print(f"t={sample.timestamp:.2f}  cpu={sample.cpu_percent:.1f}%  rss={sample.memory_rss_mb:.1f}MB")
```

Check out the ["System Metrics Monitoring" example notebook](https://github.com/awslabs/llmeter/blob/main/examples/System%20Metrics%20Monitoring.ipynb) on GitHub for a full walkthrough including load test correlation and time-series analysis.
