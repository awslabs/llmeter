# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import pickle
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from llmeter.callbacks.system_metrics import SystemMetricsMonitor, _Sample
from llmeter.results import Result


class TestSystemMetricsMonitorPickle:
    """Ensure SystemMetricsMonitor survives pickle, deepcopy, and dataclasses.asdict.

    The Runner calls dataclasses.asdict() on its _RunConfig (which includes callbacks)
    when saving run_config.json. This internally does a deepcopy of all field values.
    Callbacks with threading primitives (locks, events, threads) must handle this
    gracefully.
    """

    def test_deepcopy(self):
        """Callbacks must survive copy.deepcopy (used by dataclasses.asdict)."""
        monitor = SystemMetricsMonitor(sample_interval=0.5, per_process=True)
        monitor_copy = copy.deepcopy(monitor)

        assert monitor_copy.sample_interval == 0.5
        assert monitor_copy.per_process is True
        assert monitor_copy._stop_event is not monitor._stop_event

    def test_pickle_roundtrip(self):
        """Callbacks must survive pickle serialization."""
        monitor = SystemMetricsMonitor(sample_interval=2.0, per_process=False)
        data = pickle.dumps(monitor)
        restored = pickle.loads(data)

        assert restored.sample_interval == 2.0
        assert restored.per_process is False
        assert hasattr(restored, "_stop_event")
        assert hasattr(restored, "_thread")

    def test_asdict(self):
        """dataclasses.asdict must not raise on SystemMetricsMonitor."""
        monitor = SystemMetricsMonitor(sample_interval=1.0, per_process=True)
        d = asdict(monitor)

        assert d == {"sample_interval": 1.0, "per_process": True}

    def test_deepcopy_preserves_samples(self):
        """Collected samples should survive deepcopy."""
        monitor = SystemMetricsMonitor(sample_interval=1.0)
        monitor._samples = [
            _Sample(
                timestamp=1.0,
                cpu_percent=25.0,
                memory_rss_mb=100.0,
                memory_vms_mb=200.0,
                net_bytes_sent=1000,
                net_bytes_recv=2000,
            )
        ]
        monitor_copy = copy.deepcopy(monitor)
        assert len(monitor_copy._samples) == 1
        assert monitor_copy._samples[0].cpu_percent == 25.0


class TestSystemMetricsMonitorLifecycle:
    """Test the before_run / after_run lifecycle."""

    @pytest.fixture
    def monitor(self):
        return SystemMetricsMonitor(sample_interval=0.1, per_process=True)

    def test_before_run_starts_thread(self, monitor):
        """before_run should start a background sampling thread."""

        async def _test():
            class FakeConfig:
                pass

            await monitor.before_run(FakeConfig())
            assert monitor._thread is not None
            assert monitor._thread.is_alive()

            # Cleanup
            monitor._stop_event.set()
            monitor._thread.join(timeout=2)

        asyncio.run(_test())

    def test_after_run_stops_thread(self, monitor):
        """after_run should stop the sampling thread."""

        async def _test():
            class FakeConfig:
                pass

            await monitor.before_run(FakeConfig())
            assert monitor._thread.is_alive()

            result = Result(responses=[], total_requests=0)
            await monitor.after_run(result)

            assert monitor._thread is None or not monitor._thread.is_alive()

        asyncio.run(_test())

    def test_collects_samples(self, monitor):
        """The monitor should collect samples during the run."""

        async def _test():
            class FakeConfig:
                pass

            await monitor.before_run(FakeConfig())
            await asyncio.sleep(0.5)

            result = Result(responses=[], total_requests=0)
            await monitor.after_run(result)

            assert len(monitor._samples) >= 3

        asyncio.run(_test())

    def test_contributes_stats_to_result(self, monitor):
        """after_run should populate system stats on the result."""

        async def _test():
            class FakeConfig:
                pass

            await monitor.before_run(FakeConfig())
            await asyncio.sleep(0.5)

            result = Result(responses=[], total_requests=0)
            await monitor.after_run(result)

            stats = result.stats
            assert "system_cpu_percent-average" in stats
            assert "system_cpu_percent-p50" in stats
            assert "system_cpu_percent-p90" in stats
            assert "system_cpu_percent-p99" in stats
            assert "system_memory_rss_mb-average" in stats
            assert "system_memory_rss_mb-max" in stats
            assert "system_memory_vms_mb-average" in stats
            assert "system_memory_vms_mb-max" in stats
            assert "system_net_bytes_sent_total" in stats
            assert "system_net_bytes_recv_total" in stats
            assert "system_samples_collected" in stats
            assert stats["system_samples_collected"] >= 3
            assert stats["system_memory_rss_mb-max"] > 0

        asyncio.run(_test())

    def test_no_samples_emits_warning(self, monitor, caplog):
        """after_run with no samples should warn and not crash."""

        async def _test():
            # Don't call before_run, so no sampling thread was started
            result = Result(responses=[], total_requests=0)
            await monitor.after_run(result)

            # Should not contribute any stats
            system_keys = [k for k in result.stats if k.startswith("system_")]
            assert len(system_keys) == 0

        asyncio.run(_test())


class TestSystemMetricsMonitorPersistence:
    """Test that system metrics survive save/load round-trips."""

    def test_stats_persist_in_result_save_load(self):
        """System metrics should be preserved in stats.json across save/load."""

        async def _test():
            monitor = SystemMetricsMonitor(sample_interval=0.1, per_process=True)

            class FakeConfig:
                pass

            await monitor.before_run(FakeConfig())
            await asyncio.sleep(0.4)

            result = Result(responses=[], total_requests=0)
            await monitor.after_run(result)

            with tempfile.TemporaryDirectory() as tmpdir:
                result.output_path = tmpdir
                result.save()

                # Load with responses
                loaded = Result.load(tmpdir, load_responses=True)
                assert "system_cpu_percent-average" in loaded.stats
                assert "system_memory_rss_mb-max" in loaded.stats
                assert "system_net_bytes_sent_total" in loaded.stats

                # Load without responses (stats-only path)
                loaded_no_resp = Result.load(tmpdir, load_responses=False)
                assert "system_cpu_percent-average" in loaded_no_resp.stats
                assert "system_memory_rss_mb-max" in loaded_no_resp.stats

        asyncio.run(_test())

    def test_save_to_file_and_load(self):
        """Test callback configuration persistence."""
        monitor = SystemMetricsMonitor(sample_interval=2.5, per_process=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "monitor.json"
            monitor.save_to_file(path)

            # Verify file contents
            with open(path) as f:
                config = json.load(f)
            assert config["type"] == "SystemMetricsMonitor"
            assert config["sample_interval"] == 2.5
            assert config["per_process"] is False

            # Load back
            restored = SystemMetricsMonitor._load_from_file(path)
            assert restored.sample_interval == 2.5
            assert restored.per_process is False


class TestSystemMetricsMonitorReuse:
    """Test that a monitor can be reused across multiple runs."""

    def test_reuse_across_runs(self):
        """A monitor should reset state between runs."""

        async def _test():
            monitor = SystemMetricsMonitor(sample_interval=0.1, per_process=True)

            class FakeConfig:
                pass

            # First run
            await monitor.before_run(FakeConfig())
            await asyncio.sleep(0.3)
            result1 = Result(responses=[], total_requests=0)
            await monitor.after_run(result1)
            samples_run1 = result1.stats["system_samples_collected"]

            # Second run — should reset
            await monitor.before_run(FakeConfig())
            await asyncio.sleep(0.3)
            result2 = Result(responses=[], total_requests=0)
            await monitor.after_run(result2)
            samples_run2 = result2.stats["system_samples_collected"]

            # Both runs should have collected independent samples
            assert samples_run1 >= 2
            assert samples_run2 >= 2

        asyncio.run(_test())


class TestSystemMetricsMonitorLiveStats:
    """Test the live_stats() method for progress display integration."""

    def test_live_stats_empty_before_run(self):
        """live_stats() returns empty dict when no samples collected."""
        monitor = SystemMetricsMonitor(sample_interval=0.5)
        assert monitor.live_stats() == {}

    def test_live_stats_returns_latest_sample(self):
        """live_stats() returns current CPU and memory from latest sample."""

        async def _test():
            monitor = SystemMetricsMonitor(sample_interval=0.1, per_process=True)

            class FakeConfig:
                pass

            await monitor.before_run(FakeConfig())
            await asyncio.sleep(0.4)

            stats = monitor.live_stats()
            assert "system_cpu_percent" in stats
            assert "system_rss_mb" in stats
            assert stats["system_rss_mb"] > 0

            # Cleanup
            monitor._stop_event.set()
            monitor._thread.join(timeout=2)

        asyncio.run(_test())

    def test_live_stats_includes_network_rate(self):
        """live_stats() includes network recv rate when 2+ samples exist."""

        async def _test():
            monitor = SystemMetricsMonitor(sample_interval=0.1, per_process=True)

            class FakeConfig:
                pass

            await monitor.before_run(FakeConfig())
            await asyncio.sleep(0.4)

            stats = monitor.live_stats()
            # With multiple samples, network rate should be present
            assert "system_net_recv_kbps" in stats

            # Cleanup
            monitor._stop_event.set()
            monitor._thread.join(timeout=2)

        asyncio.run(_test())
