"""MicrobiomeBench Python API."""

from __future__ import annotations

from importlib import import_module


__version__ = "0.1.0"
__all__ = [
    "ModelBundle",
    "BatchNeutralBank",
    "BenchmarkBundle",
    "BatchEvaluation",
    "BatchSummary",
    "BenchmarkResult",
    "DAEvaluation",
    "GenerationOutput",
    "ResolvedBenchmarkScenario",
    "Simulator",
    "benchmark",
    "dataset",
    "download_bank",
    "download_dataset",
    "download_model",
    "da_result",
    "evaluate_batch",
    "evaluate_da",
    "evaluate_da_many",
    "evaluate_generation",
    "generate",
    "generate_from_condition_table",
    "import_dataset",
    "list_conditions",
    "list_datasets",
    "list_parameters",
    "list_presets",
    "load_bank",
    "load_benchmark_bundle",
    "load_model",
    "package_model",
    "prepare_data",
    "show_dataset",
    "show_preset",
    "summarize_batch",
    "summarize_da",
    "train_diffusion",
    "train_tokenizer",
    "write_config_template",
]

_PUBLIC_API = {
    "ModelBundle": (".bundle", "ModelBundle"),
    "BatchNeutralBank": (".bank", "BatchNeutralBank"),
    "BenchmarkBundle": (".benchmark_bundle", "BenchmarkBundle"),
    "BatchEvaluation": (".benchmark_evaluation", "BatchEvaluation"),
    "BatchSummary": (".benchmark_evaluation", "BatchSummary"),
    "BenchmarkResult": ("._benchmark", "BenchmarkResult"),
    "DAEvaluation": (".benchmark_evaluation", "DAEvaluation"),
    "GenerationOutput": (".generation", "GenerationOutput"),
    "ResolvedBenchmarkScenario": (".benchmark_config", "ResolvedBenchmarkScenario"),
    "Simulator": (".simulator", "Simulator"),
    "benchmark": ("._benchmark", "benchmark"),
    "dataset": (".datasets", "dataset"),
    "download_bank": (".bank", "download_bank"),
    "download_dataset": (".datasets", "download_dataset"),
    "download_model": (".models", "download_model"),
    "da_result": (".benchmark_evaluation", "da_result"),
    "evaluate_batch": (".benchmark_evaluation", "evaluate_batch"),
    "evaluate_da": (".benchmark_evaluation", "evaluate_da"),
    "evaluate_da_many": (".benchmark_evaluation", "evaluate_da_many"),
    "evaluate_generation": (".evaluate", "evaluate_generation"),
    "generate": (".generation", "generate"),
    "generate_from_condition_table": (
        ".generation",
        "generate_from_condition_table",
    ),
    "import_dataset": (".datasets", "import_dataset"),
    "list_conditions": (".models", "list_conditions"),
    "list_datasets": (".datasets", "list_datasets"),
    "list_parameters": (".cli", "list_parameters"),
    "list_presets": (".benchmark_config", "list_presets"),
    "load_bank": (".bank", "load_bank"),
    "load_benchmark_bundle": (".benchmark_bundle", "load_benchmark_bundle"),
    "load_model": (".models", "load_model"),
    "package_model": ("._package_model", "package_model"),
    "prepare_data": (".data", "prepare_data"),
    "show_dataset": (".datasets", "show_dataset"),
    "show_preset": (".benchmark_config", "show_preset"),
    "summarize_batch": (".benchmark_evaluation", "summarize_batch"),
    "summarize_da": (".benchmark_evaluation", "summarize_da"),
    "train_diffusion": ("._train_diffusion", "train_diffusion"),
    "train_tokenizer": ("._train_tokenizer", "train_tokenizer"),
    "write_config_template": (".benchmark_config", "write_config_template"),
}


def __getattr__(name: str):
    """Load an API object on first access."""

    if name not in _PUBLIC_API:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, _ = _PUBLIC_API[name]
    module = import_module(module_name, __name__)
    for public_name, (source_module, object_name) in _PUBLIC_API.items():
        if source_module == module_name:
            globals()[public_name] = getattr(module, object_name)
    return globals()[name]
