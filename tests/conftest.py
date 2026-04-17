"""Pytest configuration — custom markers and shared fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests that require network access (RCSB PDB fetches).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: marks tests requiring network access (deselected by default, "
        "use --run-network to include)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network option to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
