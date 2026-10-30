"""
Unit tests for ForgeTune v2.0 TensorShape and Dynamic Batching semantics.
"""

from forgetune import _forgetune_cpp as cpp

def test_dynamic_dim_constant():
    assert cpp.DYNAMIC_DIM == -1
    assert cpp.DEFAULT_MAX_BATCH == 128

def test_cpp_tensorshape_dynamic_batch():
    s_dynamic_2d = cpp.TensorShape([-1, 16])
    assert s_dynamic_2d.is_dynamic()
    assert s_dynamic_2d.has_dynamic_batch()
    assert s_dynamic_2d.static_element_count() == 16
    assert s_dynamic_2d.element_count(1) == 16
    assert s_dynamic_2d.element_count(8) == 128
    assert s_dynamic_2d.element_count(128) == 2048

    s_dynamic_4d = cpp.TensorShape([-1, 3, 32, 32])
    assert s_dynamic_4d.is_dynamic()
    assert s_dynamic_4d.has_dynamic_batch()
    assert s_dynamic_4d.static_element_count() == 3 * 32 * 32
    assert s_dynamic_4d.element_count(4) == 4 * 3 * 32 * 32

def test_cpp_tensorshape_static():
    s_static = cpp.TensorShape([4, 16])
    assert not s_static.is_dynamic()
    assert not s_static.has_dynamic_batch()
    assert s_static.element_count() == 64
    assert s_static.element_count(8) == 64  # Static shapes ignore dynamic_batch argument

def test_cpp_tensorshape_non_leading_dynamic_dim():
    s_invalid = cpp.TensorShape([4, -1])
    assert s_invalid.is_dynamic()
    assert not s_invalid.has_dynamic_batch()

def test_cpp_conv2d_attrs():
    attrs = cpp.Conv2DAttrs()
    attrs.stride_h = 2
    attrs.stride_w = 2
    attrs.pad_h = 1
    attrs.pad_w = 1
    attrs.dilation_h = 1
    attrs.dilation_w = 1
    attrs.groups = 1
    assert attrs.stride_h == 2
    assert attrs.pad_w == 1
