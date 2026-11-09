# mbBench

MicrobiomeBench simulates genus-level microbiome profiles and builds benchmarks with known biological and batch effects.

## Install

Requires Python 3.10–3.12.

From PyPI

```bash
python -m pip install mbbench
```

From GitHub

```bash
python -m pip install "git+https://github.com/reckonzzz/mbBench.git"
```

## Quick start

```python
import mbbench
```

**Evaluate a method.** Load a [pre-generated benchmark](https://doi.org/10.5281/zenodo.24076644) by scenario and seed. Run your method on `data.counts`, then score its results with `mbbench.evaluate_da()` or `mbbench.evaluate_batch()`.

```python
data = mbbench.dataset(preset="da-null-balanced", seed=42)
```

**Generate new profiles.** Choose `ae` (default), `vae`, or `ae-system` from the [pretrained model release](https://doi.org/10.5281/zenodo.24080099). The `ae-system` model adds Body-System conditioning.

```python
model = mbbench.download_model("ae")
result = mbbench.generate(model=model, n_samples=20, seed=42)
```

**Reuse stored logits.** Use the [core logit bank](https://doi.org/10.5281/zenodo.24082591) to generate profiles on CPU without diffusion sampling.

```python
bank = mbbench.download_bank()
result = mbbench.generate(bank=bank, n_samples=20, seed=42)
```

Use `mbbench.benchmark()` to create a benchmark and `mbbench --help` for CLI commands.

## Python API and CLI

Prefix the commands in the CLI column with `mbbench`. For full arguments, run `mbbench COMMAND --help`.

| Python API | CLI | Purpose |
| --- | --- | --- |
| `list_datasets()` | `list-datasets` | List the available pre-generated benchmark scenarios and seeds. |
| `show_dataset()` | `show-dataset` | Show the catalog entries for a scenario. |
| `download_dataset()` | `download-dataset` | Download an official benchmark dataset. The Python function also loads its tables. |
| `import_dataset()` | `import-dataset` | Install a benchmark from a local ZIP. The Python function also loads its tables. |
| `dataset()` | — | Load a benchmark by scenario and seed, from a ZIP, or from a directory. |
| `load_benchmark_bundle()` | — | Load and validate an extracted benchmark directory. |
| `download_model()` | `download-model` | Download and install an official pretrained model. The Python function returns a simulator. |
| `load_model()` | — | Load an installed official model or a custom model directory. |
| `download_bank()` | — | Download and load a pre-generated logit bank. |
| `load_bank()` | — | Load a logit bank from an extracted directory. |
| `list_conditions()` | `list-conditions` | List accepted model condition values. The CLI also accepts a local bank via `--bank`. |
| `generate()` | `generate` | Generate relative abundances or observed counts from a model or bank. |
| `generate_from_condition_table()` | `generate --condition-table PATH` | Generate samples for each row of a condition table. |
| `list_presets()` | `benchmark --list-presets` | List built-in benchmark scenarios. |
| `show_preset()` | `benchmark --show-preset NAME` | Inspect the complete configuration for a preset. |
| `list_parameters()` | `benchmark --list-parameters` | Show benchmark parameters, defaults and accepted values. |
| `write_config_template()` | `benchmark --write-config-template PATH` | Write a YAML template for a custom benchmark. |
| `benchmark()` | `benchmark` | Generate a benchmark with known truth from a preset or custom configuration. |
| `da_result()` | — | Build a complete differential-abundance decision table from significant and tested taxa. |
| `evaluate_da()` | `evaluate-da` | Score differential-abundance decisions against benchmark truth. |
| `evaluate_da_many()` | `evaluate-da --manifest PATH` | Evaluate a manifest of differential-abundance results and summarize repetitions. |
| `summarize_da()` | `summarize-da` | Aggregate differential-abundance metrics across repetitions. |
| `evaluate_batch()` | `evaluate-batch` | Score corrected abundance matrices against a benchmark. |
| `summarize_batch()` | `summarize-batch` | Aggregate batch-correction metrics and scores across repetitions. |
| `evaluate_generation()` | `evaluate` | Compare generated and reference compositions using ecological summaries and distances. |
| `prepare_data()` | `prepare` | Align count and metadata tables, normalize abundances and prepare sample splits. |
| `train_tokenizer()` | `train-tokenizer` | Train an autoencoder or variational autoencoder tokenizer. |
| `train_diffusion()` | `train-diffusion` | Train the conditional latent diffusion model. |
| `package_model()` | `package-model` | Assemble training outputs into a loadable model bundle. |
| `Simulator`, `ModelBundle` | — | A loaded simulator and its model weights and metadata. |
| `BatchNeutralBank`, `BenchmarkBundle` | — | A stored logit bank and a fully loaded benchmark dataset. |
| `GenerationOutput`, `BenchmarkResult` | — | Results returned by profile generation and benchmark generation. |
| `ResolvedBenchmarkScenario` | — | A benchmark configuration with defaults and overrides resolved. |
| `DAEvaluation`, `BatchEvaluation`, `BatchSummary` | — | Differential-abundance results, batch evaluation results and repeated batch summaries. |
| — | `--help` | Show the available commands and options. |
| — | `--version` | Show the installed package version. |
