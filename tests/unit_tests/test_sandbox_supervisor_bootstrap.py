# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failed supervisor bootstrap must remain safe to close."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from shlex import quote

import pytest

from nemo_gym.agent_utils.sandbox_session import SandboxCommand
from nemo_gym.agent_utils.supervisor_client import (
    CLEANUP_RECEIPT_FILE,
    HARNESS_COMMAND_FILE,
    LAUNCH_CLAIM_FILE,
    RUNNER_INTERPRETER,
    SUPERVISOR_FILE,
    SUPERVISOR_PID_FILE,
    SUPERVISOR_SCRIPT,
    remove_session_directory,
    stop_and_confirm_cleanup,
    supervised_launch_command,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult


class LocalSandbox:
    def __init__(self, path: str | None = None) -> None:
        self.env = os.environ | ({"PATH": path} if path is not None else {})

    async def download(self, source: str, destination: Path) -> None:
        shutil.copyfile(source, destination)

    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float = 30) -> SandboxExecResult:
        process = await asyncio.create_subprocess_exec(
            shutil.which("sh"),
            "-c",
            command,
            cwd=cwd,
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)


def _tools(tmp_path: Path, *names: str) -> str:
    """A PATH holding only the named programs."""
    directory = tmp_path / "bin"
    directory.mkdir()
    for name in names:
        (directory / name).symlink_to(shutil.which(name))
    return str(directory)


@pytest.mark.parametrize("failure", ["missing_bash", "missing_supervisor", "broken_supervisor"])
async def test_bootstrap_failure_preserves_diagnostics_and_allows_close(tmp_path: Path, failure: str) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    supervisor = session_dir / SUPERVISOR_FILE
    shutil.copyfile(SUPERVISOR_SCRIPT, supervisor)
    sandbox = LocalSandbox()
    if failure == "missing_bash":
        sandbox = LocalSandbox(_tools(tmp_path, "touch"))
        expected_error = "bash"
    elif failure == "missing_supervisor":
        supervisor.unlink()
        expected_error = SUPERVISOR_FILE
    else:
        supervisor.write_text("if then fi\n")
        expected_error = "syntax error"
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=["touch", str(session_dir / "worker-started")],
        timeout=1,
        cleanup_timeout=1,
    )
    result = await sandbox.exec(command)
    assert result.return_code != 0
    assert expected_error in result.stderr
    assert not (session_dir / LAUNCH_CLAIM_FILE).is_symlink()
    assert not (session_dir / SUPERVISOR_PID_FILE).exists()
    assert not (session_dir / "worker-started").exists()
    assert not (session_dir / CLEANUP_RECEIPT_FILE).exists()

    receipt = await stop_and_confirm_cleanup(
        LocalSandbox(), session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
    )
    assert receipt["cleanup_confirmed"] is True
    assert receipt["return_code"] is None
    assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("stop")


@pytest.mark.parametrize("removed", [False, True])
async def test_fenced_delayed_launch_does_not_require_bash(tmp_path: Path, removed: bool) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / LAUNCH_CLAIM_FILE).symlink_to("stop")
    if removed:
        shutil.rmtree(session_dir)
    result = await LocalSandbox(_tools(tmp_path, "touch")).exec(
        supervised_launch_command(
            session_dir=str(session_dir),
            command=["touch", str(tmp_path / "worker-started")],
            timeout=1,
            cleanup_timeout=1,
        )
    )
    assert result.return_code == 0
    assert result.stderr == ""
    assert not (tmp_path / "worker-started").exists()


@pytest.mark.parametrize("remove_directory", [False, True])
async def test_close_during_preflight_fences_later_launch(tmp_path: Path, remove_directory: bool) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    preflight_started = tmp_path / "preflight-started"
    continue_preflight = tmp_path / "continue-preflight"
    worker_started = tmp_path / "worker-started"
    shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
    # A bash that pauses in the syntax check stands in for a slow sandbox.
    fake = tmp_path / "fake-bin"
    fake.mkdir()
    (fake / "bash").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = -n ]; then touch {str(preflight_started)!r}; '
        f"while [ ! -e {str(continue_preflight)!r} ]; do sleep 0.01; done; exit 0; fi\n"
        f"touch {str(worker_started)!r}\n"
    )
    (fake / "bash").chmod(0o755)
    sandbox = LocalSandbox(f"{fake}:{os.environ['PATH']}")
    launch = asyncio.create_task(
        sandbox.exec(
            supervised_launch_command(session_dir=str(session_dir), command=["worker"], timeout=1, cleanup_timeout=1)
        )
    )
    try:
        async with asyncio.timeout(5):
            while not preflight_started.exists():
                await asyncio.sleep(0.01)
        receipt = await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
        )
        assert receipt["cleanup_confirmed"] is True
        assert receipt["return_code"] is None
        if remove_directory:
            await remove_session_directory(
                sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
            )
    finally:
        continue_preflight.touch()
        result = await asyncio.wait_for(launch, timeout=5)
    assert result.return_code == 0
    assert not worker_started.exists()


async def test_launch_delayed_before_linking_does_not_recreate_a_removed_session(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / SUPERVISOR_FILE).write_text("")
    resolving = tmp_path / "resolving"
    resume = tmp_path / "resume"
    python = tmp_path / "python"
    python.write_text(
        "#!/bin/sh\n"
        f'if [ "$2" = -c ]; then touch {quote(str(resolving))}; '
        f"while [ ! -e {quote(str(resume))} ]; do sleep 0.01; done; fi\n"
        f'exec {quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    sandbox = LocalSandbox()
    launch = asyncio.create_task(
        sandbox.exec(
            supervised_launch_command(
                session_dir=str(session_dir),
                command=[str(python), "worker"],
                timeout=1,
                cleanup_timeout=1,
                stdlib_runner=True,
            )
        )
    )
    try:
        async with asyncio.timeout(5):
            while not resolving.exists():
                await asyncio.sleep(0.01)
        await stop_and_confirm_cleanup(
            sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
        )
        await remove_session_directory(
            sandbox, session_dir=str(session_dir), workdir=None, timeout=1, harness="bootstrap-test"
        )
    finally:
        resume.touch()
        await asyncio.wait_for(launch, timeout=5)
    assert not session_dir.exists()


def _base_python() -> str:
    """The interpreter running these tests, outside any virtual environment."""
    return os.path.realpath(sys._base_executable)


def _harness_argv(command_file: Path) -> list[str]:
    """The NUL-separated harness argv the launch shell wrote."""
    return command_file.read_bytes().decode().split("\0")[:-1]


@pytest.mark.parametrize("stdlib_runner", [False, True])
@pytest.mark.parametrize("linkable", [True, False])
async def test_bootstrap_quotes_private_paths(tmp_path: Path, linkable: bool, stdlib_runner: bool) -> None:
    # The runner link resolves /proc/self/exe; elsewhere the launch runs the given argv.
    linkable = linkable and sys.platform == "linux"
    session_dir = tmp_path / "session's $(touch unintended) files"
    session_dir.mkdir()
    if not linkable:
        (session_dir / "bin").write_text("not a directory")  # The launch must fall back to the given argv.
    runtime = tmp_path / "runtime's $(touch unintended) files"
    runtime.mkdir()
    # Named by path and outside a virtual environment, so its link goes in the session with or without the opt-in.
    python = runtime / "python interpreter"
    python.symlink_to(_base_python())
    supervisor = session_dir / SUPERVISOR_FILE
    arguments = session_dir / "arguments"
    supervisor.write_text(f'printf "%s\\n" "$@" > {quote(str(arguments))}\n')
    command = supervised_launch_command(
        session_dir=str(session_dir),
        command=[str(python), "worker", "$(touch unintended)", "argument's value"],
        timeout=1,
        cleanup_timeout=1,
        stdlib_runner=stdlib_runner,
    )
    result = await LocalSandbox().exec(command, cwd=str(tmp_path))
    assert result.return_code == 0, result.stderr
    captured = arguments.read_text().splitlines()
    # The supervisor's command line names neither the interpreter nor the harness.
    assert "--" not in captured and not any("python" in argument for argument in captured)
    runner = session_dir / RUNNER_INTERPRETER if linkable else python
    assert _harness_argv(Path(captured[captured.index("--command-file") + 1])) == [
        str(runner),
        "worker",
        "$(touch unintended)",
        "argument's value",
    ]
    assert runner.resolve() == python.resolve()
    assert not (tmp_path / "unintended").exists()
    assert (session_dir / LAUNCH_CLAIM_FILE).readlink() == Path("launch")


@pytest.mark.parametrize("argv", [["python3"], []])
def test_stdlib_runner_rejects_an_argv_without_an_interpreter_and_runner(argv: list[str]) -> None:
    with pytest.raises(ValueError, match="stdlib_runner requires"):
        supervised_launch_command(
            session_dir="/session", command=argv, timeout=1, cleanup_timeout=1, stdlib_runner=True
        )
    with pytest.raises(ValueError, match="stdlib_runner requires"):
        SandboxCommand(argv=argv, stdlib_runner=True)


@pytest.mark.parametrize("argv", [["python3", "runner"], ["bash", "runner"], ["/runtime/bin/node", "runner"]])
def test_only_an_interpreter_named_by_path_gets_a_link_without_opting_in(argv: list[str]) -> None:
    # A python found on PATH is the task's, so its runner keeps it unless it declares a standard-library runner.
    command = supervised_launch_command(session_dir="/session", command=argv, timeout=1, cleanup_timeout=1)
    assert "gym-runner" not in command
    assert f"printf '%s\\0' {' '.join(argv)} >/session/harness.argv" in command
    assert SandboxCommand(argv=["/runtime/venv/bin/python", "runner"]).stdlib_runner is False


def _launch_shape(tmp_path: Path, shape: str, report: Path) -> list[str]:
    """A harness argv in one of the adapters' launch shapes; its runner reports what it runs as, then waits."""
    runner = tmp_path / "runner"  # Uploaded without a .py suffix, as the adapters do.
    runner.write_text(
        "import json, os, sys, time\n"
        "try:\n"
        "    import marker\n"
        "    where = marker.WHERE\n"
        "except ImportError:\n"
        "    where = None\n"
        "comm = open('/proc/self/comm').read().strip()\n"
        f"open({str(report)!r} + '.part', 'w').write(json.dumps("
        "{'pid': os.getpid(), 'prefix': os.path.realpath(sys.prefix), 'comm': comm, 'marker': where}))\n"
        f"os.replace({str(report)!r} + '.part', {str(report)!r})\n"
        "time.sleep(60)\n"
    )
    if shape == "venv":  # Hermes and Stirrup: <runtime>/venv/bin/python runner ...
        venv = tmp_path / "runtime" / "venv"
        subprocess.run([_base_python(), "-m", "venv", "--without-pip", str(venv)], check=True)
        site = subprocess.run(
            [str(venv / "bin" / "python"), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        Path(site, "marker.py").write_text("WHERE = 'venv'\n")
        return [str(venv / "bin" / "python"), str(runner), "input.json"]
    runtime = tmp_path / "runtime" / "bin"
    runtime.mkdir(parents=True)
    if shape == "wrapper":  # Hermes sandboxed: a hermes-python wrapper that execs the runtime's binary.
        python = runtime / "rt-python"
        python.write_text(f'#!/bin/sh\nexec {quote(_base_python())} "$@"\n')
        python.chmod(0o755)
    else:  # A standalone runtime outside any venv.
        python = runtime / "python3"
        python.symlink_to(_base_python())
    if shape == "module":  # NOOA: <runtime>/python/bin/python3 -I -m <module> ...
        return [str(python), "-I", "-m", "cProfile", str(runner)]
    return [str(python), "-I", str(runner), "request.json"]


def _session_processes(session_dir: Path) -> set[int]:
    """Every live process whose command line names the session directory."""
    found = set()
    for entry in Path("/proc").iterdir():
        try:
            if entry.name.isdigit() and str(session_dir) in (entry / "cmdline").read_bytes().decode(errors="replace"):
                found.add(int(entry.name))
        except OSError:
            continue
    return found


async def _wait_for_report(report: Path) -> dict:
    for _ in range(500):
        if report.exists():
            return json.loads(report.read_text())
        await asyncio.sleep(0.02)
    raise AssertionError("the runner never reported")


# Test names stay free of interpreter names: pytest builds tmp_path from them, and the session path is on every
# command line.
@pytest.mark.skipif(sys.platform != "linux" or shutil.which("pgrep") is None, reason="needs Linux /proc and pgrep")
@pytest.mark.parametrize("shape", ["venv", "standalone", "wrapper", "module"])
async def test_interpreter_name_kills_miss_the_supervisor_and_a_runtime_runner(tmp_path: Path, shape: str) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
    report = session_dir / "report.json"
    argv = _launch_shape(tmp_path, shape, report)
    command = supervised_launch_command(session_dir=str(session_dir), command=argv, timeout=30, cleanup_timeout=1)
    launch = asyncio.create_task(LocalSandbox().exec(command, cwd=str(tmp_path), timeout_s=60))
    try:
        reported = await _wait_for_report(report)
        # The runner keeps its environment and packages under a neutral name.
        if shape == "venv":
            venv = tmp_path / "runtime" / "venv"
            assert (reported["prefix"], reported["marker"]) == (os.path.realpath(venv), "venv")
            assert (venv / "bin" / "gym-runner").is_symlink()
            assert not (session_dir / RUNNER_INTERPRETER).exists()
        else:
            assert (reported["prefix"], reported["marker"]) == (os.path.realpath(sys.base_prefix), None)
        assert reported["comm"] == "gym-runner"
        processes = _session_processes(session_dir) | {reported["pid"]}
        assert len(processes) >= 2  # The supervisor and the runner, at least.
        # pgrep sees the runner under its new name.
        named = subprocess.run(["pgrep", "-x", "gym-runner"], capture_output=True, text=True).stdout.split()
        assert reported["pid"] in map(int, named)
        # pgrep selects exactly as pkill and killall do, without signalling anything.
        for pattern in (["python"], ["python3"], ["-f", "python"], ["-f", r"\.py"], ["-x", "python3"]):
            matched = subprocess.run(["pgrep", *pattern], capture_output=True, text=True).stdout.split()
            assert not set(map(int, matched)) & processes, pattern
    finally:
        receipt = await stop_and_confirm_cleanup(
            LocalSandbox(), session_dir=str(session_dir), workdir=None, timeout=5, harness="kill-test"
        )
        await asyncio.wait_for(launch, timeout=30)
    assert receipt["cleanup_confirmed"] is True


@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux /proc")
async def test_launches_sharing_a_venv_each_run_under_its_link(tmp_path: Path) -> None:
    # Every launch replaces the venv's link in one rename, so concurrent launches never see a missing or partial one.
    argv = _launch_shape(tmp_path, "venv", tmp_path / "unused")
    venv = tmp_path / "runtime" / "venv"
    sessions = []
    for index in range(8):
        session_dir = tmp_path / f"session-{index}"
        session_dir.mkdir()
        shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
        report = session_dir / "report.json"
        (tmp_path / f"runner-{index}").write_text(
            (tmp_path / "runner").read_text().replace(str(tmp_path / "unused"), str(report))
        )
        command = supervised_launch_command(
            session_dir=str(session_dir),
            command=[argv[0], str(tmp_path / f"runner-{index}")],
            timeout=30,
            cleanup_timeout=1,
        )
        sessions.append((session_dir, report, command))
    launches = [
        asyncio.create_task(LocalSandbox().exec(command, cwd=str(tmp_path), timeout_s=60))
        for _, _, command in sessions
    ]
    try:
        for _, report, _ in sessions:
            reported = await _wait_for_report(report)
            assert (reported["prefix"], reported["marker"], reported["comm"]) == (
                os.path.realpath(venv),
                "venv",
                "gym-runner",
            )
        assert sorted(path.name for path in (venv / "bin").glob("gym-runner*")) == ["gym-runner"]
    finally:
        for session_dir, _, _ in sessions:
            await stop_and_confirm_cleanup(
                LocalSandbox(), session_dir=str(session_dir), workdir=None, timeout=5, harness="kill-test"
            )
        await asyncio.wait_for(asyncio.gather(*launches), timeout=30)


@pytest.mark.skipif(sys.platform != "linux" or os.geteuid() == 0, reason="needs Linux and a user that file modes bind")
async def test_a_venv_that_cannot_be_written_runs_the_argv_unchanged(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    shutil.copyfile(SUPERVISOR_SCRIPT, session_dir / SUPERVISOR_FILE)
    report = session_dir / "report.json"
    argv = _launch_shape(tmp_path, "venv", report)
    venv_bin = tmp_path / "runtime" / "venv" / "bin"
    venv_bin.chmod(0o555)
    try:
        command = supervised_launch_command(session_dir=str(session_dir), command=argv, timeout=30, cleanup_timeout=1)
        launch = asyncio.create_task(LocalSandbox().exec(command, cwd=str(tmp_path), timeout_s=60))
        try:
            reported = await _wait_for_report(report)
            # Still the venv's interpreter and packages, under its own name.
            assert (reported["prefix"], reported["marker"]) == (os.path.realpath(venv_bin.parent), "venv")
            assert reported["comm"] == "python"
            assert not (venv_bin / "gym-runner").exists()
            assert _harness_argv(session_dir / HARNESS_COMMAND_FILE) == argv
        finally:
            await stop_and_confirm_cleanup(
                LocalSandbox(), session_dir=str(session_dir), workdir=None, timeout=5, harness="kill-test"
            )
            await asyncio.wait_for(launch, timeout=30)
    finally:
        venv_bin.chmod(0o755)
