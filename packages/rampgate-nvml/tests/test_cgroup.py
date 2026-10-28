"""`pid_to_container_id` on fixture ``/proc/<pid>/cgroup`` files (no GPU, no containers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rampgate_nvml import container_id_from_cgroup, pid_to_container_id

FIXTURES = Path(__file__).parent / "fixtures" / "cgroup"

A = "dfb507a4ac3f8b56aa2cf4e8c07fb4a845b4f072dd6e313f1bb6aaf24b655ced"
B = "4f1c2e9a7b3d5c8e0f2a4b6c8d0e2f4a6b8c0d2e4f6a8b0c2d4e6f8a0b2c4d6e"
C = "0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9"


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        # Captured on the development machine (Docker 29, cgroup v2, systemd driver).
        ("v2_docker_systemd", A),
        # The same container read from a container with --pid=host and a private cgroupns.
        ("v2_docker_systemd_private_cgroupns", A),
        ("v2_docker_cgroupfs", B),
        ("v1_docker_cgroupfs", B),
        ("v1_docker_systemd", A),
        ("v2_containerd_k8s", C),
        ("v1_containerd_k8s", C),
        ("v2_crio", C),
        ("v2_podman_rootful", B),
        ("v2_podman_rootless", B),
        ("v2_docker_in_docker", B),  # the innermost container
        ("v2_podman_conmon", None),  # the container monitor, not the container
        ("v2_host_process", None),
        ("v1_host_process", None),
    ],
)
def test_fixture(fixture: str, expected: str | None) -> None:
    text = (FIXTURES / f"{fixture}.txt").read_text()
    assert container_id_from_cgroup(text) == expected


def test_pid_lookup_reads_proc_root(tmp_path: Path) -> None:
    (tmp_path / "4242").mkdir()
    (tmp_path / "4242" / "cgroup").write_text((FIXTURES / "v2_docker_systemd.txt").read_text())
    assert pid_to_container_id(4242, proc_root=tmp_path) == A
    assert pid_to_container_id(4242, proc_root=str(tmp_path)) == A


def test_missing_pid_is_none(tmp_path: Path) -> None:
    assert pid_to_container_id(1, proc_root=tmp_path) is None


def test_garbage_lines_are_skipped() -> None:
    assert container_id_from_cgroup("") is None
    assert container_id_from_cgroup("not a cgroup line\n") is None
    assert container_id_from_cgroup(f"garbage\n0::/docker/{B}\n") == B
    # A 64-character segment that is not hex is not an id.
    assert container_id_from_cgroup("0::/docker/" + "g" * 64) is None
