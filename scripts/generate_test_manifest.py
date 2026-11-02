"""
Generates the authoritative ForgeTune v2.0 Test Manifest (reports/v2_test_manifest.json).
Executes all collected Python tests + 13 native C++ tests and captures:
- Exact test nodeid
- Execution outcome (PASSED, BLOCKED/FAILED, ERROR, COLLECTION_ERROR)
- Error exception class and diagnostic message
- Grouping by directory and module
- Environment, Git commit, branch, and clean-tree metadata
"""

import os
import sys
import json
import time
import subprocess
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

class TestResultCollector:
    def __init__(self):
        self.results = {}
        self.total_collected = 0

    def pytest_collection_finish(self, session):
        self.total_collected = len(session.items)

    def pytest_collectreport(self, report):
        if report.failed:
            exc_repr = str(report.longrepr)
            self.results[report.nodeid or f"collection_error_{len(self.results)}"] = {
                "outcome": "collection_error",
                "exception_type": "CollectionError",
                "details": exc_repr.splitlines()[-1] if exc_repr else "Failed to collect test file"
            }

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.results[report.nodeid] = {
                "outcome": "skipped",
                "details": str(report.longrepr) if report.longrepr else "Skipped"
            }
        elif report.when == "call":
            if report.passed:
                self.results[report.nodeid] = {"outcome": "passed"}
            elif report.failed:
                exc_repr = str(report.longrepr) if report.longrepr else ""
                exc_type = "UnknownError"
                if hasattr(report.longrepr, "reprcrash") and report.longrepr.reprcrash:
                    crash_msg = str(report.longrepr.reprcrash.message or "")
                    if ":" in crash_msg:
                        first = crash_msg.split(":", 1)[0].strip()
                        if first and " " not in first:
                            exc_type = first
                if exc_type == "UnknownError" and exc_repr:
                    for line in reversed(exc_repr.splitlines()):
                        line = line.strip()
                        if ":" in line and not line.startswith("E "):
                            token = line.split(":", 1)[0].strip()
                            if token.endswith("Error") or token.endswith("Exception"):
                                exc_type = token
                                break
                        elif line.startswith("E ") and ":" in line:
                            token = line[2:].split(":", 1)[0].strip()
                            if token.endswith("Error") or token.endswith("Exception"):
                                exc_type = token
                                break

                if "4551" in exc_repr or "ERROR_APP_CONTROL_BLOCKED" in exc_repr:
                    exc_type = "WinError 4551 (Windows Smart App Control Block)"

                self.results[report.nodeid] = {
                    "outcome": "failed",
                    "exception_type": exc_type,
                    "details": exc_repr.splitlines()[-1] if exc_repr else ""
                }
        elif report.when in ("setup", "teardown") and report.failed:
            exc_repr = str(report.longrepr) if report.longrepr else ""
            exc_type = "SetupError" if report.when == "setup" else "TeardownError"
            if "4551" in exc_repr or "ERROR_APP_CONTROL_BLOCKED" in exc_repr:
                exc_type = "WinError 4551 (Windows Smart App Control Block)"
            self.results[report.nodeid] = {
                "outcome": "error",
                "phase": report.when,
                "exception_type": exc_type,
                "details": exc_repr.splitlines()[-1] if exc_repr else ""
            }

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    collector = TestResultCollector()
    print("Running Pytest suite to generate authoritative manifest...")
    pytest_rc = pytest.main(["tests/", "-q", "--tb=line"], plugins=[collector])

    # Run native C++ test suite dynamically in-process via _forgetune_cpp
    from forgetune import _forgetune_cpp as cpp
    if not hasattr(cpp, "run_cpp_test_suite_detailed"):
        raise RuntimeError("Detailed native test registry unavailable. Rebuild the native extension before generating the manifest.")

    cpp_results = cpp.run_cpp_test_suite_detailed()
    cpp_tests_count = len(cpp_results)
    cpp_passed_count = sum(1 for _, passed in cpp_results if passed)
    cpp_test_cases = [
        {"name": name, "status": "passed" if passed else "failed"}
        for name, passed in cpp_results
    ]
    cpp_res = 0 if cpp_passed_count == cpp_tests_count else 1

    # Reconciled execution accounting: executed includes passed, failed, and errors
    executed_py_tests = sum(1 for r in collector.results.values() if r["outcome"] in ("passed", "failed", "error"))
    total_py_collected = collector.total_collected if collector.total_collected > 0 else len(collector.results)
    passed_py = sum(1 for r in collector.results.values() if r["outcome"] == "passed")
    skipped_py = sum(1 for r in collector.results.values() if r["outcome"] == "skipped")
    failed_py = sum(1 for r in collector.results.values() if r["outcome"] == "failed")
    error_py = sum(1 for r in collector.results.values() if r["outcome"] == "error")
    coll_err_py = sum(1 for r in collector.results.values() if r["outcome"] == "collection_error")

    # Reconciliation check: collected items = passed + failed + skipped + errors (collection errors recorded separately)
    assert total_py_collected == (passed_py + skipped_py + failed_py + error_py), \
        f"Manifest test accounting mismatch: collected {total_py_collected} != passed ({passed_py}) + skipped ({skipped_py}) + failed ({failed_py}) + error ({error_py})"

    git_sha = "unknown"
    git_branch = "unknown"
    git_clean = False
    source_clean = True
    dirty_files = []
    generated_report_files = {
        "reports/v2_test_manifest.json",
        "reports/v2_paired_b64_benchmark_data.json",
        "reports/v2_benchmark_report.json",
        "reports/V2_BENCHMARK_REPORT.md",
        "reports/V2_FINAL_VERIFICATION.md",
        "reports/V2_MEMORY_REPORT.md",
        "reports/V2_FINAL_CLOSURE_AUDIT.md",
        "reports/junit_report.xml",
        "reports/junit_linux.xml",
    }
    try:
        r_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if r_sha.returncode == 0:
            git_sha = r_sha.stdout.strip()
        r_br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        if r_br.returncode == 0:
            git_branch = r_br.stdout.strip()
        r_st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if r_st.returncode == 0:
            for line in r_st.stdout.splitlines():
                line = line.strip()
                if line:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        fn = parts[1].replace("\\", "/")
                        dirty_files.append(fn)
                        if fn not in generated_report_files and not any(fn.startswith(p) for p in ("build/", ".pytest_cache/", ".venv/", "bin/")):
                            source_clean = False
            git_clean = (len(dirty_files) == 0)
    except Exception:
        pass

    manifest = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "os_platform": sys.platform,
            "python_version": sys.version,
            "git_commit": git_sha,
            "git_branch": git_branch,
            "working_tree_clean": git_clean,
            "source_files_clean": source_clean,
            "dirty_files": dirty_files,
            "harness_description": f"ForgeTune Authoritative Test Manifest ({total_py_collected} collected pytest items + {cpp_tests_count} native C++ unit tests)"
        },
        "summary": {
            "total_python_tests_collected": total_py_collected,
            "total_python_tests_executed": executed_py_tests,
            "python_passed": passed_py,
            "python_skipped": skipped_py,
            "python_failed": failed_py,
            "python_error": error_py,
            "python_collection_error": coll_err_py,
            "native_cpp_tests": cpp_tests_count,
            "native_cpp_passed": cpp_passed_count,
            "total_tests_available": total_py_collected + cpp_tests_count,
            "total_passing": passed_py + cpp_passed_count,
            "total_blocked_by_sac": sum(1 for r in collector.results.values() if "WinError 4551" in str(r.get("exception_type", "")) or "4551" in str(r.get("details", ""))),
            "pytest_return_code": int(pytest_rc)
        },
        "native_cpp_test_cases": cpp_test_cases,
        "python_tests": collector.results
    }

    os.makedirs("reports", exist_ok=True)
    manifest_path = "reports/v2_test_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=======================================================")
    print(" AUTHORITATIVE TEST MANIFEST GENERATION SUMMARY")
    print("=======================================================")
    print(f" Python Tests Collected:   {manifest['summary']['total_python_tests_collected']}")
    print(f"   - Passed:               {manifest['summary']['python_passed']}")
    print(f"   - Skipped:              {manifest['summary']['python_skipped']}")
    print(f"   - Failed:               {manifest['summary']['python_failed']}")
    print(f"   - Error:                {manifest['summary']['python_error']}")
    print(f" Native C++ Tests:         {manifest['summary']['native_cpp_tests']}")
    print(f"   - Passed:               {manifest['summary']['native_cpp_passed']}")
    print(f" Total Tests:              {manifest['summary']['total_tests_available']}")
    print(f" Total Passing:            {manifest['summary']['total_passing']}")
    print(f" Total Blocked by SAC:     {manifest['summary']['total_blocked_by_sac']}")
    print(f" Pytest Return Code:       {manifest['summary']['pytest_return_code']}")
    print(f" Saved to:                 {manifest_path}")

    has_collection_error = (coll_err_py > 0)
    has_error = (error_py > 0)
    has_failed = (failed_py > 0)

    APPROVED_LINUX_SKIP_NODE = "tests/correctness/test_audit_regressions.py::test_regression_locked_dll_raises_informative_compilation_error"
    allowed_skips = {APPROVED_LINUX_SKIP_NODE} if sys.platform != "win32" else set()
    unapproved_skips = [
        nid for nid, r in collector.results.items()
        if r.get("outcome") == "skipped" and not any(nid.replace("\\", "/").strip() == allowed or nid.replace("\\", "/").strip().endswith(allowed) for allowed in allowed_skips)
    ]
    has_unapproved_skip = len(unapproved_skips) > 0
    if has_unapproved_skip:
        print(f"[-] Unapproved test skips detected ({len(unapproved_skips)}):")
        for s in unapproved_skips:
            print(f"    - {s}")

    if pytest_rc != 0 or cpp_res != 0 or has_collection_error or has_error or has_failed or has_unapproved_skip:
        print(f"[-] Test manifest generation completed with errors/failures/unapproved skips (pytest_rc={pytest_rc}, cpp_res={cpp_res})")
        sys.exit(int(pytest_rc) if pytest_rc != 0 else 1)

if __name__ == "__main__":
    main()
