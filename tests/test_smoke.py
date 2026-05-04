"""Smoke tests for the DECOMP pipeline. Run with `pytest tests/`."""

import importlib


def test_decomp_imports():
    importlib.import_module("decomp")
    importlib.import_module("decomp.data")
    importlib.import_module("decomp.glm")
    importlib.import_module("decomp.svca")
    importlib.import_module("decomp.cca")
    importlib.import_module("decomp.viz")
    importlib.import_module("decomp.pipeline")


def test_third_party_stack():
    importlib.import_module("one.api")
    importlib.import_module("brainwidemap")
    importlib.import_module("neurencoding.linear")
    importlib.import_module("neuropop.dimensionality")
