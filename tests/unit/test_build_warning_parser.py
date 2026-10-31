"""
Unit tests for the robust compiler warning parser in scripts/build_native.py.
Verifies correct detection of relative, POSIX, Windows drive-letter, Clang, and MSVC diagnostic formats.
"""

from scripts.build_native import is_codebase_warning

def test_warning_parser_relative_path_clang():
    assert is_codebase_warning("src/codegen.cpp:45:12: warning: unused variable 'x'")
    assert is_codebase_warning("include/forgetune/ir.hpp:100: warning: comparison between signed and unsigned")
    assert is_codebase_warning("tests/cpp/test_main.cpp:25: warning: implicit conversion")

def test_warning_parser_windows_drive_letter_clang():
    line = r"D:\a\Forgetune\Forgetune\src\codegen.cpp:45:10: warning: overflow in arithmetic"
    assert is_codebase_warning(line)

    line2 = r"C:\Users\runner\Desktop\Forgetune\include\forgetune\memory_planner.hpp:20: warning: uninitialized"
    assert is_codebase_warning(line2)

def test_warning_parser_windows_msvc_format():
    # MSVC warning C4244 format with parentheses line numbers
    line_msvc = r"D:\a\Forgetune\src\codegen.cpp(42): warning C4244: 'conversion' from 'int' to 'float'"
    assert is_codebase_warning(line_msvc)

    line_rel = r"src\fusion.cpp(88): warning C4018: '<': signed/unsigned mismatch"
    assert is_codebase_warning(line_rel)

def test_warning_parser_posix_absolute_path():
    line = "/home/runner/work/Forgetune/src/pass_manager.cpp:120: warning: unused parameter"
    assert is_codebase_warning(line)

def test_warning_parser_suppresses_external_toolchain_warnings():
    # Zig libcxx internal headers
    assert not is_codebase_warning(r"C:\hostedtoolcache\zig\libcxx\include\memory:12: warning: nullability")
    assert not is_codebase_warning("/opt/hostedtoolcache/zig/libcxx/include/string:5: warning: nullability")
    # Python/Torch site-packages headers
    assert not is_codebase_warning(r"D:\a\Forgetune\.venv\Lib\site-packages\torch\include\c10\api.h(10): warning C4000: deprecated")
    assert not is_codebase_warning("/usr/include/python3.12/pyport.h:10: warning: nullability")

def test_warning_parser_fails_closed_on_unplaced_warning():
    # An unplaced warning with no file location should fail closed (return True to alert developer)
    assert is_codebase_warning("warning: cannot find entry point")
    assert is_codebase_warning("warning C9999: internal toolchain state unknown")

def test_warning_parser_ignores_non_warning_lines():
    assert not is_codebase_warning("Compiling src/codegen.cpp...")
    assert not is_codebase_warning("Generated 133 files successfully.")
    assert not is_codebase_warning("note: see previous declaration of 'foo'")
