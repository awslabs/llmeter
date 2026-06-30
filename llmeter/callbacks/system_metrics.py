# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""System resource monitoring callback for LLMeter runs."""

import logging
import threading
import time
from dataclasses import dataclass

from upath.types import ReadablePathLike, WritablePathLike

from ..results import Result
from ..runner import _RunConfig
from ..utils import DeferredError, summary_stats_from_list
from .base import Callback

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError as e:
    logger.debug(
        "psutil not available. System metrics monitoring requires psutil. "
        "Install with: pip install 'llmeter[system-metrics]'"
    )
    psutil = DeferredError(e)


@dataclass
class _Sample:
    """A single system metrics sample."""

    timestamp: float
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    net_bytes_sent: int
    net_bytes_recv: int


@dataclass
class SystemMetricsMonitor(Callback):
    """Monitor system resources (CPU, memory, network I/O) during a benchmark run.

    This callback spawns a background thread that periodically samples system metrics
    while the benchmark is running. After the run completes, aggregated statistics are
    contributed to the Result.

    Args:
        sample_interval: Seconds between samples. Default 1.0.
        per_process: If True, monitor only the current process. If False, monitor
            system-wide metrics. Default True.

    Note:
        When ``per_process=True``, CPU and memory metrics are scoped to the current Python
        process. However, **network I/O is always system-wide** because ``psutil`` does not
        support per-process network counters on most platforms. If other processes on the
        machine are generating significant network traffic during a benchmark, that activity
        will be reflected in the network stats regardless of this setting.

    Contributed stats keys (prefixed with ``system_``):

    - ``system_cpu_percent-average``, ``-p50``, ``-p90``, ``-p99``: CPU usage percentage
    - ``system_memory_rss_mb-average``, ``-p50``, ``-p90``, ``-p99``, ``-max``: Resident memory (MB)
    - ``system_memory_vms_mb-average``, ``-p50``, ``-p90``, ``-p99``, ``-max``: Virtual memory (MB)
    - ``system_net_bytes_sent_total``: Total bytes sent during the run
    - ``system_net_bytes_recv_total``: Total bytes received during the run
    - ``system_net_bytes_sent_per_second-average``: Average send rate (bytes/s)
    - ``system_net_bytes_recv_per_second-average``: Average receive rate (bytes/s)
    - ``system_samples_collected``: Number of samples taken

    Example::

        from llmeter.callbacks.system_metrics import SystemMetricsMonitor
        from llmeter.runner import Runner

        monitor = SystemMetricsMonitor(sample_interval=0.5, per_process=True)
        runner = Runner(endpoint=endpoint, callbacks=[monitor])
        result = await runner.run(payload=payload)

        print(f"CPU avg: {result.stats['system_cpu_percent-average']:.1f}%")
        print(f"Memory peak: {result.stats['system_memory_rss_mb-max']:.1f} MB")
        print(f"Network sent: {result.stats['system_net_bytes_sent_total']} bytes")
    """

    sample_interval: float = 1.0
    per_process: bool = True

    def __post_init__(self):
        self._samples: list[_Sample] = []
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._process: object = None
        self._net_start: tuple[int, int] = (0, 0)

    def __getstate__(self):
        """Support pickling/deepcopy by excluding non-serializable thread state."""
        return {
            "sample_interval": self.sample_interval,
            "per_process": self.per_process,
            "_samples": self._samples,
            "_net_start": self._net_start,
        }

    def __setstate__(self, state):
        """Restore from pickle/deepcopy, reinitializing thread primitives."""
        self.sample_interval = state["sample_interval"]
        self.per_process = state["per_process"]
        self._samples = state["_samples"]
        self._net_start = state["_net_start"]
        self._thread = None
        self._stop_event = threading.Event()
        self._process = None

    def _collect_sample(self) -> _Sample:
        """Collect a single metrics sample."""
        now = time.perf_counter()

        if self.per_process:
            cpu = self._process.cpu_percent(interval=None)
            mem_info = self._process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)
        else:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            rss_mb = mem.used / (1024 * 1024)
            vms_mb = mem.total / (1024 * 1024)

        net = psutil.net_io_counters()
        return _Sample(
            timestamp=now,
            cpu_percent=cpu,
            memory_rss_mb=rss_mb,
            memory_vms_mb=vms_mb,
            net_bytes_sent=net.bytes_sent,
            net_bytes_recv=net.bytes_recv,
        )

    def _sampling_loop(self):
        """Background thread loop that collects samples at the configured interval."""
        # Initial CPU reading to prime the counter (psutil requires two calls for %)
        if self.per_process:
            self._process.cpu_percent(interval=None)
        else:
            psutil.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            try:
                sample = self._collect_sample()
                self._samples.append(sample)
            except Exception as e:
                logger.debug(f"System metrics sampling error: {e}")
            self._stop_event.wait(self.sample_interval)

    async def before_run(self, run_config: _RunConfig) -> None:
        """Start the background sampling thread."""
        # Reset state from any previous run
        self._samples = []
        self._stop_event.clear()

        # Initialize process handle
        if self.per_process:
            self._process = psutil.Process()

        # Record starting network counters
        net = psutil.net_io_counters()
        self._net_start = (net.bytes_sent, net.bytes_recv)

        # Start sampling thread
        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="llmeter-system-metrics",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "System metrics monitoring started (interval=%.2fs, per_process=%s)",
            self.sample_interval,
            self.per_process,
        )

    async def after_run(self, result: Result) -> None:
        """Stop sampling and contribute aggregated stats to the result."""
        # Stop the background thread
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        if not self._samples:
            logger.warning("No system metrics samples were collected during the run.")
            return

        # Aggregate metrics
        stats = self._compute_stats()

        # Contribute to result
        result._update_contributed_stats(stats)

        logger.info(
            "System metrics: %d samples collected. CPU avg=%.1f%%, "
            "Memory RSS peak=%.1f MB, Net sent=%d bytes, Net recv=%d bytes",
            len(self._samples),
            stats.get("system_cpu_percent-average", 0),
            stats.get("system_memory_rss_mb-max", 0),
            stats.get("system_net_bytes_sent_total", 0),
            stats.get("system_net_bytes_recv_total", 0),
        )

    def _compute_stats(self) -> dict[str, float | int]:
        """Compute aggregated statistics from collected samples."""
        stats: dict[str, float | int] = {}

        if not self._samples:
            return stats

        # CPU percent
        cpu_values = [s.cpu_percent for s in self._samples]
        cpu_stats = summary_stats_from_list(cpu_values)
        for agg, value in cpu_stats.items():
            stats[f"system_cpu_percent-{agg}"] = value

        # Memory RSS (MB)
        rss_values = [s.memory_rss_mb for s in self._samples]
        rss_stats = summary_stats_from_list(rss_values)
        for agg, value in rss_stats.items():
            stats[f"system_memory_rss_mb-{agg}"] = value
        stats["system_memory_rss_mb-max"] = max(rss_values)

        # Memory VMS (MB)
        vms_values = [s.memory_vms_mb for s in self._samples]
        vms_stats = summary_stats_from_list(vms_values)
        for agg, value in vms_stats.items():
            stats[f"system_memory_vms_mb-{agg}"] = value
        stats["system_memory_vms_mb-max"] = max(vms_values)

        # Network I/O totals (delta from start to last sample)
        last_sample = self._samples[-1]
        stats["system_net_bytes_sent_total"] = (
            last_sample.net_bytes_sent - self._net_start[0]
        )
        stats["system_net_bytes_recv_total"] = (
            last_sample.net_bytes_recv - self._net_start[1]
        )

        # Per-interval network rates (average, p50, p90, p99)
        if len(self._samples) >= 2:
            send_rates = []
            recv_rates = []
            for i in range(1, len(self._samples)):
                dt = self._samples[i].timestamp - self._samples[i - 1].timestamp
                if dt > 0:
                    send_rates.append(
                        (
                            self._samples[i].net_bytes_sent
                            - self._samples[i - 1].net_bytes_sent
                        )
                        / dt
                    )
                    recv_rates.append(
                        (
                            self._samples[i].net_bytes_recv
                            - self._samples[i - 1].net_bytes_recv
                        )
                        / dt
                    )
            if send_rates:
                send_rate_stats = summary_stats_from_list(send_rates)
                for agg, value in send_rate_stats.items():
                    stats[f"system_net_bytes_sent_per_second-{agg}"] = value
            if recv_rates:
                recv_rate_stats = summary_stats_from_list(recv_rates)
                for agg, value in recv_rate_stats.items():
                    stats[f"system_net_bytes_recv_per_second-{agg}"] = value

        stats["system_samples_collected"] = len(self._samples)

        return stats

    @property
    def samples(self) -> list[_Sample]:
        """Access the raw collected samples for custom analysis."""
        return self._samples

    def live_stats(self) -> dict[str, float | int]:
        """Return the latest system metrics for the live progress display.

        This method is called by the Runner's refresh loop to include system
        metrics in the live stats table during a run. It returns the most recent
        sample's values (not aggregates), providing a real-time view.

        Returns:
            A dict with the latest CPU, memory, and network rate values, or
            an empty dict if no samples have been collected yet.
        """
        if not self._samples:
            return {}

        latest = self._samples[-1]
        stats: dict[str, float | int] = {
            "system_cpu_percent": latest.cpu_percent,
            "system_rss_mb": round(latest.memory_rss_mb, 1),
        }

        # Include network rate if we have at least 2 samples
        if len(self._samples) >= 2:
            prev = self._samples[-2]
            dt = latest.timestamp - prev.timestamp
            if dt > 0:
                stats["system_net_recv_kbps"] = round(
                    (latest.net_bytes_recv - prev.net_bytes_recv) / dt / 1024, 1
                )

        return stats

    def save_to_file(self, path: WritablePathLike) -> None:
        """Save this SystemMetricsMonitor configuration to file."""
        import json

        from ..utils import ensure_path

        out_path = ensure_path(path)
        config = {
            "type": "SystemMetricsMonitor",
            "sample_interval": self.sample_interval,
            "per_process": self.per_process,
        }
        with out_path.open("w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def _load_from_file(cls, path: ReadablePathLike) -> "SystemMetricsMonitor":
        """Load a SystemMetricsMonitor from file."""
        import json

        from ..utils import ensure_path

        in_path = ensure_path(path)
        with in_path.open("r") as f:
            config = json.load(f)

        return cls(
            sample_interval=config.get("sample_interval", 1.0),
            per_process=config.get("per_process", True),
        )
