from __future__ import annotations

import argparse
from importlib.resources import files
import math
from pathlib import Path

import yaml

from . import __version__
from .defaults import DEFAULT_MAX_IN_MEMORY_MB


GENERATION_DEFAULTS = yaml.safe_load(
    files("mbbench.defaults").joinpath("generation.yaml").read_text(encoding="utf-8")
)


class DefaultHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Display meaningful defaults while omitting unset optional values."""

    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


class HelpParser(argparse.ArgumentParser):
    """Argument parser that displays defaults for every command."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", DefaultHelpFormatter)
        super().__init__(*args, **kwargs)


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model",
        choices=("ae", "vae", "custom"),
        help="Use an official AE/VAE model or a packaged custom model; default: ae.",
    )
    source.add_argument(
        "--bank",
        help="Batch-neutral logit bank directory; runs generation on CPU.",
    )
    parser.add_argument(
        "--model-dir",
        help="Custom model bundle directory; required and valid only with --model custom.",
    )
    parser.add_argument(
        "--use-body-system",
        action="store_true",
        help="Use the official AE Body-System model or a compatible custom model.",
    )


def _benchmark_parameter(
    option: str,
    yaml_path: str,
    help_text: str,
    accepted: str,
    *,
    common: bool = False,
    **argument_options,
) -> dict[str, object]:
    return {
        "option": option,
        "dest": option.removeprefix("--").replace("-", "_"),
        "yaml_path": yaml_path,
        "help": help_text,
        "accepted": accepted,
        "common": common,
        "argument_options": argument_options,
    }


BENCHMARK_PARAMETERS = (
    _benchmark_parameter(
        "--model",
        "model.type",
        "Model family.",
        "ae, vae, or custom",
        common=True,
        choices=("ae", "vae", "custom"),
    ),
    _benchmark_parameter(
        "--model-dir",
        "model.directory",
        "Custom model bundle directory.",
        "path; only with --model custom",
        common=True,
    ),
    _benchmark_parameter(
        "--guidance-scale",
        "model.guidance_scale",
        "Classifier-free guidance scale.",
        "finite number >= 0",
        common=True,
        type=float,
    ),
    _benchmark_parameter(
        "--use-body-system",
        "model.use_body_system",
        "Enable or disable Body-System conditioning.",
        "--use-body-system or --no-use-body-system",
        action=argparse.BooleanOptionalAction,
    ),
    _benchmark_parameter(
        "--common-condition",
        "conditions.common",
        "Shared model condition; repeat for multiple fields.",
        "FIELD=VALUE; unique fields",
        common=True,
        action="append",
        metavar="FIELD=VALUE",
    ),
    _benchmark_parameter(
        "--group",
        "conditions.groups",
        "Replace preset groups; repeat once per group.",
        "LABEL[,FIELD=VALUE...]; at least two unique labels",
        common=True,
        action="append",
        metavar="LABEL[,FIELD=VALUE...]",
    ),
    _benchmark_parameter(
        "--body-site",
        "conditions.common.Body_Site_Coarse",
        "Shared Body-Site condition; a convenience form of --common-condition.",
        "nonempty model value",
        common=True,
    ),
    _benchmark_parameter(
        "--n-samples",
        "design.n_samples",
        "Total benchmark samples.",
        "positive integer satisfying all minimum cell counts",
        common=True,
        type=int,
    ),
    _benchmark_parameter(
        "--n-batches",
        "design.n_batches",
        "Number of simulated batches.",
        "positive integer",
        common=True,
        type=int,
    ),
    _benchmark_parameter(
        "--design-mode",
        "design.mode",
        "Group-by-batch allocation.",
        "balanced or confounded",
        choices=("balanced", "confounded"),
    ),
    _benchmark_parameter(
        "--case-fractions",
        "design.case_fractions",
        "Case fraction in each batch for binary confounding.",
        "one value in (0, 1) per batch; binary confounded design only; exclusive with --group-fractions",
        common=True,
        type=float,
        nargs="+",
    ),
    _benchmark_parameter(
        "--group-fractions",
        "design.group_fractions",
        "One multiclass fraction row per batch; repeat for every batch.",
        "comma-separated positive fractions summing to 1; multiclass confounded design only; exclusive with --case-fractions",
        action="append",
        metavar="F1,F2,...",
    ),
    _benchmark_parameter(
        "--minimum-cell-count",
        "design.minimum_cell_count",
        "Minimum samples per group-by-batch cell.",
        "positive integer",
        type=int,
    ),
    _benchmark_parameter(
        "--biology-mode",
        "biology.mode",
        "Biological effect construction.",
        "null, strict, abundance, prevalence, mixed, or natural",
        common=True,
        choices=("null", "strict", "abundance", "prevalence", "mixed", "natural"),
    ),
    _benchmark_parameter(
        "--n-targets",
        "biology.n_targets",
        "Number of automatically selected biological targets.",
        "positive integer; even in strict mode; exclusive with --target-taxa; artificial biology only",
        common=True,
        type=int,
    ),
    _benchmark_parameter(
        "--target-taxa",
        "biology.target_taxa",
        "Comma-separated biological target taxa; replaces automatic selection.",
        "unique comma-separated taxa; exclusive with --n-targets and automatic prevalence bounds; artificial biology only",
        common=True,
    ),
    _benchmark_parameter(
        "--target-prevalence-min",
        "biology.prevalence_min",
        "Automatic target prevalence lower bound.",
        "0 < min < max < 1; automatic target selection only",
        type=float,
    ),
    _benchmark_parameter(
        "--target-prevalence-max",
        "biology.prevalence_max",
        "Automatic target prevalence upper bound.",
        "0 < min < max < 1; automatic target selection only",
        type=float,
    ),
    _benchmark_parameter(
        "--effect-strength",
        "biology.effect_strength",
        "Biological log-scale effect strength.",
        "finite number > 0 when targets are enabled",
        common=True,
        type=float,
    ),
    _benchmark_parameter(
        "--effect-directions",
        "biology.effect_directions",
        "Biological target directions.",
        "balanced, random-balanced, or increase",
        choices=("balanced", "random-balanced", "increase"),
    ),
    _benchmark_parameter(
        "--batch-mechanisms",
        "batch.mechanisms",
        "Enabled batch-effect mechanisms.",
        "zero or more of global, targeted_abundance, prevalence, depth, dispersion",
        common=True,
        choices=(
            "global",
            "targeted_abundance",
            "prevalence",
            "depth",
            "dispersion",
        ),
        nargs="*",
    ),
    _benchmark_parameter(
        "--global-batch-strength",
        "batch.global_strength",
        "Global composition batch strength.",
        "finite number > 0; requires global in --batch-mechanisms",
        type=float,
    ),
    _benchmark_parameter(
        "--batch-target-fraction",
        "batch.target_fraction",
        "Fraction of eligible targeted-abundance taxa.",
        "value in (0, 1]; exclusive with --batch-target-taxa",
        type=float,
    ),
    _benchmark_parameter(
        "--batch-target-taxa",
        "batch.target_taxa",
        "Comma-separated targeted-abundance taxa.",
        "unique taxa; exclusive with --batch-target-fraction",
    ),
    _benchmark_parameter(
        "--batch-target-strength",
        "batch.target_strength",
        "Targeted-abundance batch strength.",
        "finite number > 0; requires targeted_abundance in --batch-mechanisms",
        type=float,
    ),
    _benchmark_parameter(
        "--batch-prevalence-fraction",
        "batch.prevalence_fraction",
        "Fraction of eligible prevalence targets.",
        "value in (0, 1]; exclusive with --batch-prevalence-taxa",
        type=float,
    ),
    _benchmark_parameter(
        "--batch-prevalence-taxa",
        "batch.prevalence_taxa",
        "Comma-separated prevalence targets.",
        "unique taxa; exclusive with --batch-prevalence-fraction",
    ),
    _benchmark_parameter(
        "--batch-prevalence-strength",
        "batch.prevalence_strength",
        "Prevalence batch strength.",
        "finite number > 0; requires prevalence in --batch-mechanisms",
        type=float,
    ),
    _benchmark_parameter(
        "--batch-depth-strength",
        "batch.depth_strength",
        "Log-depth batch strength.",
        "finite number > 0; requires depth in --batch-mechanisms",
        type=float,
    ),
    _benchmark_parameter(
        "--batch-dispersion-strength",
        "batch.dispersion_strength",
        "Log-dispersion batch strength.",
        "finite number > 0; requires dispersion in --batch-mechanisms and dirichlet-multinomial observation",
        type=float,
    ),
    _benchmark_parameter(
        "--allow-target-overlap",
        "batch.allow_target_overlap",
        "Allow biological and targeted batch taxa to overlap.",
        "Boolean; requires direct biological targets and targeted_abundance or prevalence",
        action=argparse.BooleanOptionalAction,
    ),
    _benchmark_parameter(
        "--support-mode",
        "support.mode",
        "Zero/nonzero support decoder.",
        "deterministic, bernoulli, or temperature-bernoulli",
        choices=("deterministic", "bernoulli", "temperature-bernoulli"),
    ),
    _benchmark_parameter(
        "--support-threshold",
        "support.threshold",
        "Support probability threshold.",
        "value in (0, 1); deterministic or temperature-bernoulli mode only",
        type=float,
    ),
    _benchmark_parameter(
        "--support-temperature",
        "support.temperature",
        "Temperature for stochastic support decoding.",
        "finite number > 0; temperature-bernoulli mode only",
        type=float,
    ),
    _benchmark_parameter(
        "--observation-model",
        "observation.model",
        "Count observation model.",
        "multinomial or dirichlet-multinomial",
        choices=("multinomial", "dirichlet-multinomial"),
    ),
    _benchmark_parameter(
        "--depth-mode",
        "observation.depth_mode",
        "Fixed or model-donor library depths.",
        "fixed or model-donor",
        choices=("fixed", "model-donor"),
    ),
    _benchmark_parameter(
        "--library-size",
        "observation.library_size",
        "Library size in fixed-depth mode.",
        "positive integer; fixed depth mode only",
        type=int,
    ),
    _benchmark_parameter(
        "--depth-min",
        "observation.depth_min",
        "Minimum observed library depth.",
        "positive integer <= depth-max",
        type=int,
    ),
    _benchmark_parameter(
        "--depth-max",
        "observation.depth_max",
        "Maximum observed library depth.",
        "positive integer >= depth-min",
        type=int,
    ),
    _benchmark_parameter(
        "--concentration-mode",
        "observation.concentration_mode",
        "Dirichlet-multinomial concentration distribution.",
        "fixed or lognormal; dirichlet-multinomial observation only",
        choices=("fixed", "lognormal"),
    ),
    _benchmark_parameter(
        "--concentration-median",
        "observation.concentration_median",
        "Median or fixed concentration.",
        "positive number within concentration bounds; dirichlet-multinomial observation only",
        type=float,
    ),
    _benchmark_parameter(
        "--concentration-log-sd",
        "observation.concentration_log_sd",
        "Lognormal concentration standard deviation.",
        "finite number > 0; lognormal concentration mode only",
        type=float,
    ),
    _benchmark_parameter(
        "--concentration-min",
        "observation.concentration_min",
        "Minimum concentration.",
        "0 < min <= median; dirichlet-multinomial observation only",
        type=float,
    ),
    _benchmark_parameter(
        "--concentration-max",
        "observation.concentration_max",
        "Maximum concentration.",
        "max >= median; dirichlet-multinomial observation only",
        type=float,
    ),
    _benchmark_parameter(
        "--paired-neutral-counts",
        "observation.paired_neutral_counts",
        "Also observe paired counts from the batch-neutral reference.",
        "--paired-neutral-counts or --no-paired-neutral-counts",
        action=argparse.BooleanOptionalAction,
    ),
    _benchmark_parameter(
        "--seed", "runtime.seed", "Benchmark seed.", "integer", common=True, type=int
    ),
    _benchmark_parameter(
        "--batch-size",
        "runtime.batch_size",
        "Model generation chunk size.",
        "positive integer",
        common=True,
        type=int,
    ),
    _benchmark_parameter(
        "--device",
        "runtime.device",
        "PyTorch device such as auto, cpu, cuda, or cuda:0.",
        "auto or installed PyTorch device",
        common=True,
    ),
)

BENCHMARK_OVERRIDE_PATHS = {
    str(parameter["dest"]): str(parameter["yaml_path"])
    for parameter in BENCHMARK_PARAMETERS
}


def _benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--preset", help="Named benchmark preset; dedicated options may override it."
    )
    source.add_argument(
        "--config",
        help="Complete custom YAML; cannot be combined with scientific overrides.",
    )
    parser.add_argument(
        "--bank",
        help="Batch-neutral logit bank directory; cannot be combined with model overrides.",
    )
    parser.add_argument(
        "--list-presets", action="store_true", help="List available presets and exit."
    )
    parser.add_argument(
        "--show-preset", metavar="NAME", help="Print one complete preset YAML and exit."
    )
    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="List every CLI override and YAML path, then exit.",
    )
    parser.add_argument(
        "--write-config-template",
        metavar="PATH",
        help="Write a complete custom YAML template and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without generating data.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for generated data, truth tables, and resolved benchmark.yaml.",
    )
    parser.add_argument(
        "--max-in-memory-mb",
        type=float,
        default=DEFAULT_MAX_IN_MEMORY_MB,
        help="Maximum estimated memory before benchmark generation streams to output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a nonempty output directory after staging succeeds.",
    )

    for parameter in BENCHMARK_PARAMETERS:
        action = parser.add_argument(
            str(parameter["option"]),
            default=argparse.SUPPRESS,
            help=str(parameter["help"]),
            **dict(parameter["argument_options"]),
        )
        action.mbbench_yaml_path = parameter["yaml_path"]
        action.mbbench_accepted = parameter["accepted"]
    visible = {
        "help",
        "preset",
        "config",
        "list_presets",
        "show_preset",
        "list_parameters",
        "write_config_template",
        "dry_run",
        "output_dir",
        "max_in_memory_mb",
        "overwrite",
        "bank",
        *(
            str(parameter["dest"])
            for parameter in BENCHMARK_PARAMETERS
            if parameter["common"]
        ),
    }
    for action in parser._actions:
        if action.dest not in visible:
            action.mbbench_parameter_help = action.help
            action.help = argparse.SUPPRESS


def build_parser() -> argparse.ArgumentParser:
    parser = HelpParser(prog="mbbench", description="MicrobiomeBench")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=HelpParser
    )

    listing = commands.add_parser(
        "list-conditions", help="List model condition levels."
    )
    _model_arguments(listing)
    listing.add_argument(
        "--phenotypes",
        action="store_true",
        help="List supported phenotype values only.",
    )
    listing.add_argument(
        "--body-sites",
        action="store_true",
        help="List supported Body-Site values only.",
    )
    listing.add_argument(
        "--field",
        action="append",
        help="Condition field to list; repeat for multiple fields.",
    )

    download = commands.add_parser("download-model", help="Download an official model.")
    download.add_argument(
        "--model", choices=("ae", "vae"), default="ae", help="Official model family."
    )
    download.add_argument(
        "--use-body-system",
        action="store_true",
        help="Download the official AE Body-System model.",
    )
    download.add_argument("--url", help="Override the registered official archive URL.")
    download.add_argument(
        "--install-dir",
        help="Installation directory; the interactive default is the mbBench model cache.",
    )

    dataset_list = commands.add_parser(
        "list-datasets", help="List official pre-generated benchmark datasets."
    )
    dataset_list.add_argument(
        "--dataset-version",
        default="latest",
        help="Dataset release to inspect.",
    )

    dataset_show = commands.add_parser(
        "show-dataset", help="Show one pre-generated benchmark dataset."
    )
    dataset_show.add_argument("preset", help="Benchmark preset name.")
    dataset_show.add_argument(
        "--dataset-version",
        default="latest",
        help="Dataset release to inspect.",
    )

    dataset_download = commands.add_parser(
        "download-dataset",
        help="Download and install a pre-generated benchmark dataset.",
    )
    dataset_download.add_argument("preset", help="Benchmark preset name.")
    dataset_download.add_argument(
        "--seed", type=int, default=42, help="Pre-generated dataset seed."
    )
    dataset_download.add_argument(
        "--dataset-version",
        default="latest",
        help="Dataset release to download.",
    )
    dataset_download.add_argument(
        "--install-dir", help="Installation directory; defaults to the dataset cache."
    )
    dataset_download.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing installed dataset.",
    )

    dataset_import = commands.add_parser(
        "import-dataset", help="Install a manually downloaded dataset ZIP."
    )
    dataset_import.add_argument("archive", help="Local dataset ZIP archive.")
    dataset_import.add_argument(
        "--install-dir", help="Installation directory; defaults to the dataset cache."
    )
    dataset_import.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing installed dataset.",
    )

    prepare = commands.add_parser("prepare", help="Prepare training data.")
    prepare.add_argument(
        "--abundance",
        required=True,
        help="Sample-by-taxon count table (CSV or CSV.GZ).",
    )
    prepare.add_argument(
        "--metadata",
        required=True,
        help="Sample metadata CSV aligned by sample identifier.",
    )
    prepare.add_argument(
        "--output-dir", required=True, help="Directory for prepared training data."
    )
    prepare.add_argument(
        "--sample-column",
        default="Sample",
        help="Sample identifier column in both input tables.",
    )
    prepare.add_argument(
        "--phenotype-column",
        default="Phenotype_Tokens",
        help="Standardized phenotype column.",
    )
    prepare.add_argument(
        "--body-site-column",
        default="Body_Site_Coarse",
        help="Standardized Body-Site column.",
    )
    split = prepare.add_mutually_exclusive_group()
    split.add_argument(
        "--split-column",
        help="Metadata column containing train, validation, and test labels.",
    )
    split.add_argument(
        "--split-file",
        help="JSON split lists, or a CSV containing sample IDs and a split column.",
    )
    prepare.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when mbBench creates a sample-level split.",
    )
    prepare.add_argument(
        "--body-system-map",
        help="CSV mapping Body-Site values to derived Body-System values.",
    )
    prepare.add_argument(
        "--taxonomy",
        help="Optional taxonomy JSON, or a CSV containing taxon and family columns.",
    )
    prepare.add_argument(
        "--biological-fields",
        nargs="+",
        metavar="FIELD",
        help="Biological condition fields; defaults to Phenotype_Tokens and Body_Site_Coarse.",
    )
    multilabel = prepare.add_mutually_exclusive_group()
    multilabel.add_argument(
        "--multilabel-fields",
        nargs="+",
        metavar="FIELD",
        help="Fields containing multiple tokens; defaults to Phenotype_Tokens.",
    )
    multilabel.add_argument(
        "--no-multilabel-fields",
        dest="multilabel_fields",
        action="store_const",
        const=[],
        help="Treat every selected biological field as a single categorical value.",
    )
    technical = prepare.add_mutually_exclusive_group()
    technical.add_argument(
        "--technical-fields",
        nargs="+",
        metavar="FIELD",
        help=(
            "Technical covariate fields; defaults to Instrument, Variable_Region, "
            "Sequencing_Type, and Library_Layout."
        ),
    )
    technical.add_argument(
        "--no-technical-fields",
        dest="technical_fields",
        action="store_const",
        const=[],
        help="Prepare data without technical covariates.",
    )
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files in a nonempty output directory.",
    )

    tokenizer = commands.add_parser(
        "train-tokenizer",
        help="Train an AE or VAE tokenizer.",
        description="Train an AE or VAE tokenizer. Priority: CLI options > --config YAML > built-in defaults.",
    )
    tokenizer.add_argument(
        "--model",
        choices=("ae", "vae"),
        default="ae",
        help="Tokenizer family to train.",
    )
    tokenizer.add_argument(
        "--config",
        help="YAML configuration; CLI values override YAML, then built-in defaults.",
    )
    tokenizer.add_argument(
        "--data-dir",
        required=True,
        help="Prepared data directory produced by mbbench prepare.",
    )
    tokenizer.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints, progress, loss history, and resolved training settings.",
    )
    tokenizer.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as auto, cpu, cuda, or cuda:0.",
    )
    tokenizer.add_argument(
        "--seed", type=int, help="Training seed; otherwise use the selected YAML value."
    )
    tokenizer.add_argument(
        "--epochs",
        type=int,
        help="Maximum epochs; otherwise use the selected YAML value.",
    )
    tokenizer.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size; otherwise use the selected YAML value.",
    )
    tokenizer.add_argument(
        "--learning-rate",
        type=float,
        help="Initial learning rate; otherwise use the selected YAML value.",
    )
    tokenizer.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Epochs without improvement before stopping.",
    )
    tokenizer.add_argument(
        "--early-stopping-min-delta",
        type=float,
        help="Minimum validation-loss change counted as improvement.",
    )
    tokenizer.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        help="Minimum epochs completed before early stopping.",
    )
    tokenizer_run = tokenizer.add_mutually_exclusive_group()
    tokenizer_run.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last.pt in the output directory.",
    )
    tokenizer_run.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a new run in a nonempty output directory.",
    )

    diffusion = commands.add_parser(
        "train-diffusion",
        help="Train conditional latent diffusion.",
        description="Train conditional latent diffusion. Priority: CLI options > --config YAML > built-in defaults.",
    )
    diffusion.add_argument(
        "--config",
        help="YAML configuration; CLI values override YAML, then built-in defaults.",
    )
    diffusion.add_argument(
        "--data-dir",
        required=True,
        help="Prepared data directory produced by mbbench prepare.",
    )
    diffusion.add_argument(
        "--tokenizer-checkpoint",
        required=True,
        help="Trained tokenizer best.pt checkpoint.",
    )
    diffusion.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints, progress, loss history, and resolved training settings.",
    )
    diffusion.add_argument(
        "--use-body-system",
        action="store_true",
        help="Add Body-System derived from the prepared mapping.",
    )
    diffusion.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as auto, cpu, cuda, or cuda:0.",
    )
    diffusion.add_argument(
        "--seed", type=int, help="Training seed; otherwise use the selected YAML value."
    )
    diffusion.add_argument(
        "--epochs",
        type=int,
        help="Maximum epochs; otherwise use the selected YAML value.",
    )
    diffusion.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size; otherwise use the selected YAML value.",
    )
    diffusion.add_argument(
        "--learning-rate",
        type=float,
        help="Initial learning rate; otherwise use the selected YAML value.",
    )
    diffusion.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Epochs without improvement before stopping.",
    )
    diffusion.add_argument(
        "--early-stopping-min-delta",
        type=float,
        help="Minimum validation-loss change counted as improvement.",
    )
    diffusion.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        help="Minimum epochs completed before early stopping.",
    )
    diffusion_run = diffusion.add_mutually_exclusive_group()
    diffusion_run.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last.pt in the output directory.",
    )
    diffusion_run.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a new run in a nonempty output directory.",
    )

    package = commands.add_parser(
        "package-model", help="Assemble a trained custom model."
    )
    package.add_argument(
        "--tokenizer-checkpoint",
        required=True,
        help="Trained tokenizer best.pt checkpoint.",
    )
    package.add_argument(
        "--diffusion-checkpoint",
        required=True,
        help="Trained diffusion best.pt checkpoint.",
    )
    package.add_argument(
        "--data-dir",
        required=True,
        help="Prepared data used to train both checkpoints.",
    )
    package.add_argument(
        "--output-dir", required=True, help="Destination custom model bundle directory."
    )
    package.add_argument(
        "--use-body-system",
        action="store_true",
        help="Record derived Body-System conditioning in the bundle.",
    )
    package.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files in a nonempty output directory.",
    )

    generate = commands.add_parser("generate", help="Generate microbiome samples.")
    _model_arguments(generate)
    generate.add_argument(
        "--phenotype",
        default=GENERATION_DEFAULTS["phenotype"],
        help="Phenotype for direct generation.",
    )
    generate.add_argument(
        "--body-site",
        default=GENERATION_DEFAULTS["body_site"],
        help="Body-Site for direct generation.",
    )
    generation_conditions = generate.add_mutually_exclusive_group()
    generation_conditions.add_argument(
        "--condition-table",
        help="CSV with one condition combination per row and an optional n_samples column.",
    )
    generation_conditions.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Additional condition; repeat for multiple fields.",
    )
    generate.add_argument(
        "--n-samples",
        type=int,
        default=GENERATION_DEFAULTS["n_samples"],
        help="Samples per direct condition or CSV row without n_samples.",
    )
    generate.add_argument(
        "--batch-size",
        type=int,
        default=GENERATION_DEFAULTS["batch_size"],
        help="Maximum samples generated in one model call.",
    )
    generate.add_argument(
        "--max-in-memory-mb",
        type=float,
        default=DEFAULT_MAX_IN_MEMORY_MB,
        help="Maximum estimated memory before generation streams directly to output.",
    )
    generate.add_argument(
        "--output-mode",
        choices=("relative", "observed"),
        default=GENERATION_DEFAULTS["output_mode"],
        help="Write batch-neutral relative abundance or observation-layer integer counts.",
    )
    generate.add_argument(
        "--guidance-scale",
        type=float,
        default=GENERATION_DEFAULTS["guidance_scale"],
        help="Classifier-free guidance scale.",
    )
    generate.add_argument(
        "--support-temperature",
        type=float,
        default=GENERATION_DEFAULTS["support_temperature"],
        help="Temperature for stochastic taxon support decoding.",
    )
    generate.add_argument(
        "--library-size",
        type=int,
        help="Fixed read count per sample in observed mode; otherwise sample learned depths.",
    )
    generate.add_argument(
        "--seed", type=int, default=GENERATION_DEFAULTS["seed"], help="Generation seed."
    )
    generate.add_argument(
        "--device",
        default=GENERATION_DEFAULTS["device"],
        help="PyTorch device such as auto, cpu, cuda, or cuda:0.",
    )
    generate.add_argument(
        "--output-dir",
        required=True,
        help="Directory for abundance, metadata, optional counts, and resolved generation settings.",
    )
    generate.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files in a nonempty output directory.",
    )

    benchmark = commands.add_parser(
        "benchmark",
        help="Generate a truth-informed benchmark from a preset or complete YAML.",
        description=(
            "Use --preset with optional dedicated overrides, or use one complete --config YAML. "
            "The two modes cannot be mixed."
        ),
    )
    _benchmark_arguments(benchmark)

    evaluate = commands.add_parser("evaluate", help="Evaluate generated data.")
    evaluate.add_argument(
        "--real-abundance",
        required=True,
        help="Reference nonnegative count or abundance CSV; rows are normalized internally.",
    )
    evaluate.add_argument(
        "--synthetic-abundance",
        required=True,
        help="Generated count or abundance CSV with matching taxa; rows are normalized internally.",
    )
    evaluate.add_argument(
        "--output-dir",
        required=True,
        help="Directory for compact diversity and distance metrics.",
    )
    evaluate.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files in a nonempty output directory.",
    )

    evaluate_batch = commands.add_parser(
        "evaluate-batch",
        help="Evaluate corrected abundance matrices against a benchmark.",
    )
    evaluate_batch.add_argument(
        "--benchmark", required=True, help="Extracted mbBench benchmark directory."
    )
    evaluate_batch.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        metavar="METHOD=CSV",
        help="Corrected sample-by-taxon abundance matrix; repeat for multiple methods.",
    )
    evaluate_batch.add_argument(
        "--prevalence-min",
        type=float,
        default=0.10,
        help="Observed prevalence filter for structure metrics and optional scores.",
    )
    evaluate_batch.add_argument(
        "--maximum-relative-abundance-min",
        type=float,
        default=0.002,
        help="Observed maximum relative-abundance filter for structure metrics and optional scores.",
    )
    evaluate_batch.add_argument(
        "--calculate-scores",
        action="store_true",
        help="Also calculate the separate publication component and overall score table.",
    )
    evaluate_batch.add_argument(
        "--output-dir", required=True, help="Directory for batch evaluation tables."
    )
    evaluate_batch.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a nonempty evaluation output directory.",
    )

    evaluate_da = commands.add_parser(
        "evaluate-da", help="Evaluate taxon-level DA decisions against benchmark truth."
    )
    evaluate_da.add_argument(
        "--benchmark", help="Extracted benchmark directory for a single evaluation."
    )
    evaluate_da.add_argument(
        "--input",
        dest="inputs",
        action="append",
        metavar="METHOD=CSV",
        help="Taxon-level DA decision table; repeat for multiple methods.",
    )
    evaluate_da.add_argument(
        "--manifest",
        help="CSV with benchmark, input, and method columns for many repetitions.",
    )
    evaluate_da.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
        help="Bootstrap repetitions for a manifest summary.",
    )
    evaluate_da.add_argument(
        "--summary-seed",
        type=int,
        default=42,
        help="Bootstrap seed for a manifest summary.",
    )
    evaluate_da.add_argument(
        "--output-dir", required=True, help="Directory for DA evaluation tables."
    )
    evaluate_da.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a nonempty evaluation output directory.",
    )

    summarize_batch = commands.add_parser(
        "summarize-batch", help="Summarize repetition-level batch evaluations."
    )
    summarize_batch.add_argument(
        "--input", required=True, help="Batch metrics CSV or directory to scan."
    )
    summarize_batch.add_argument(
        "--output-dir", required=True, help="Directory for repeated-run summaries."
    )
    summarize_batch.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a nonempty summary output directory.",
    )

    summarize_da = commands.add_parser(
        "summarize-da", help="Summarize DA repetitions and empirical FDR."
    )
    summarize_da.add_argument(
        "--input", required=True, help="DA metrics CSV or directory to scan."
    )
    summarize_da.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
        help="Bootstrap repetitions for each method, scenario, and metric.",
    )
    summarize_da.add_argument(
        "--seed", type=int, default=42, help="Base seed for stable bootstrap streams."
    )
    summarize_da.add_argument(
        "--output-dir", required=True, help="Directory for repeated-run summaries."
    )
    summarize_da.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a nonempty summary output directory.",
    )
    return parser


def validate_model_arguments(args: argparse.Namespace) -> None:
    bank = getattr(args, "bank", None)
    model = getattr(args, "model", None)
    if bank and (
        model is not None
        or getattr(args, "model_dir", None)
        or getattr(args, "use_body_system", False)
    ):
        raise SystemExit(
            "--bank cannot be combined with --model, --model-dir, or --use-body-system."
        )
    if model is None:
        if getattr(args, "model_dir", None):
            raise SystemExit("--model-dir requires --model custom.")
        return
    if model == "custom" and not getattr(args, "model_dir", None):
        raise SystemExit("--model custom requires --model-dir.")
    if model != "custom" and getattr(args, "model_dir", None):
        raise SystemExit("--model-dir is valid only with --model custom.")
    if model == "vae" and getattr(args, "use_body_system", False):
        raise SystemExit(
            "The official VAE does not include Body-System; use a compatible custom model."
        )


def _comma_values(value: str, option: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise SystemExit(f"{option} requires unique comma-separated nonempty values.")
    return values


def _key_values(items: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{option} values must use FIELD=VALUE.")
        field, value = (part.strip() for part in item.split("=", 1))
        if not field or not value or field in result:
            raise SystemExit(
                f"{option} requires unique fields and nonempty FIELD=VALUE entries."
            )
        result[field] = value
    return result


def _groups(items: list[str]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for item in items:
        pieces = [piece.strip() for piece in item.split(",") if piece.strip()]
        if not pieces or "=" in pieces[0]:
            raise SystemExit("--group values must use LABEL[,FIELD=VALUE...].")
        label = pieces[0]
        conditions = _key_values(pieces[1:], "--group") if len(pieces) > 1 else {}
        groups.append({"label": label, "conditions": conditions})
    if len(groups) < 2 or len({str(group["label"]) for group in groups}) != len(groups):
        raise SystemExit(
            "--group must be repeated for at least two uniquely labelled groups."
        )
    return groups


def _benchmark_overrides(args: argparse.Namespace) -> dict[str, object]:
    supplied = vars(args)
    overrides = {
        path: supplied[name]
        for name, path in BENCHMARK_OVERRIDE_PATHS.items()
        if name in supplied
    }
    for name in ("target_taxa", "batch_target_taxa", "batch_prevalence_taxa"):
        if name in supplied:
            overrides[BENCHMARK_OVERRIDE_PATHS[name]] = _comma_values(
                supplied[name], f"--{name.replace('_', '-')}"
            )
    if "group_fractions" in supplied:
        try:
            overrides["design.group_fractions"] = [
                [float(value.strip()) for value in row.split(",") if value.strip()]
                for row in supplied["group_fractions"]
            ]
        except ValueError as error:
            raise SystemExit(
                "--group-fractions rows must contain comma-separated numbers."
            ) from error
    if "common_condition" in supplied:
        common = _key_values(supplied["common_condition"], "--common-condition")
        if "body_site" in supplied and "Body_Site_Coarse" in common:
            raise SystemExit(
                "Do not provide Body_Site_Coarse in --common-condition together with --body-site."
            )
        overrides.pop("conditions.common", None)
        for field, value in common.items():
            overrides[f"conditions.common.{field}"] = value
    if "group" in supplied:
        overrides["conditions.groups"] = _groups(supplied["group"])
    return overrides


def _print_benchmark_parameters(
    parser: argparse.ArgumentParser, preset: str | None
) -> None:
    import json

    del parser
    reference_name = preset or "da-strict-common-moderate"
    print(f"Defaults shown below come from preset {reference_name!r}.")
    print(
        "Use --list-parameters together with --preset NAME to inspect another preset."
    )
    current_category = None
    for parameter in list_parameters(reference_name):
        category = str(parameter["category"])
        if category != current_category:
            current_category = category
            print(f"\n[{category}]")
            print("option\tYAML path\tdefault\taccepted values\tmeaning")
        print(
            f"{parameter['option']}\t{parameter['yaml_path']}\t"
            f"{json.dumps(parameter['default'], ensure_ascii=False, separators=(',', ':'))}\t"
            f"{parameter['accepted']}\t{parameter['description']}"
        )


def list_parameters(
    preset: str | None = None,
) -> list[dict[str, object]]:
    """List benchmark parameters, defaults, constraints, and YAML paths."""

    from .benchmark_config import resolve_preset

    reference_name = preset or "da-strict-common-moderate"
    defaults = resolve_preset(reference_name).as_dict()

    def value_at(path: str) -> object:
        value: object = defaults
        for part in path.split("."):
            value = value[part]  # type: ignore[index]
        return value

    category_order = {
        name: index
        for index, name in enumerate(
            (
                "model",
                "conditions",
                "design",
                "biology",
                "batch",
                "support",
                "observation",
                "runtime",
            )
        )
    }
    entries = sorted(
        BENCHMARK_PARAMETERS,
        key=lambda parameter: category_order[
            str(parameter["yaml_path"]).split(".", 1)[0]
        ],
    )
    rows = []
    for parameter in entries:
        option = str(parameter["option"])
        if (
            parameter["argument_options"].get("action")
            is argparse.BooleanOptionalAction
        ):
            option = f"{option}/--no-{option.removeprefix('--')}"
        path = str(parameter["yaml_path"])
        rows.append(
            {
                "category": path.split(".", 1)[0],
                "option": option,
                "yaml_path": path,
                "default": value_at(path),
                "accepted": str(parameter["accepted"]),
                "description": str(parameter["help"]),
            }
        )
    return rows


def _positive_sample_count(value: object) -> int:
    if isinstance(value, bool):
        raise SystemExit("Every n_samples value must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit("Every n_samples value must be a positive integer.") from error
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise SystemExit("Every n_samples value must be a positive integer.")
    return int(numeric)


def _run_generate(args: argparse.Namespace) -> Path:
    import csv

    import pandas as pd

    from .generation import _generate_from_condition_table
    from .pretrained import official_model_name, resolve_model_directory
    from .simulator import Simulator

    if args.bank:
        from .bank import load_bank

        root = Path(args.bank).expanduser().resolve()
        simulator = load_bank(root)
        selected_model = None
        use_body_system = None
    else:
        selected_model = args.model or GENERATION_DEFAULTS["model"]
        root = resolve_model_directory(
            selected_model, args.model_dir, args.use_body_system
        )
        simulator = Simulator.from_directory(root, args.device)
        use_body_system = (
            simulator.bundle.supports_body_system
            if selected_model == "custom"
            else args.use_body_system
        )
    direct_conditions = _key_values(args.condition, "--condition")
    reserved = sorted(
        set(direct_conditions)
        & {"phenotype", "Phenotype_Tokens", "body_site", "Body_Site_Coarse"}
    )
    if reserved:
        raise SystemExit(
            "Use --phenotype and --body-site instead of repeating standard fields in --condition: "
            + ", ".join(reserved)
            + "."
        )
    if args.condition_table:
        source = Path(args.condition_table).expanduser().resolve()
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        duplicates = sorted({field for field in header if header.count(field) > 1})
        if duplicates:
            raise SystemExit(
                f"Conditions CSV contains duplicate columns: {duplicates}."
            )
        alias_conflicts = [
            (alias, canonical)
            for alias, canonical in (
                ("phenotype", "Phenotype_Tokens"),
                ("body_site", "Body_Site_Coarse"),
            )
            if alias in header and canonical in header
        ]
        if alias_conflicts:
            conflicts = ", ".join(
                f"{alias}/{canonical}" for alias, canonical in alias_conflicts
            )
            raise SystemExit(
                f"Conditions CSV repeats equivalent condition fields: {conflicts}."
            )
        conditions = pd.read_csv(source)
        if conditions.empty:
            raise SystemExit("Condition table contains no rows.")
        condition_fields = set(simulator.diffusion_config.primary_fields)
        allowed_columns = {"n_samples", *condition_fields}
        if "Phenotype_Tokens" in condition_fields:
            allowed_columns.add("phenotype")
        if "Body_Site_Coarse" in condition_fields:
            allowed_columns.add("body_site")
        unknown_columns = sorted(set(conditions.columns) - allowed_columns)
        if unknown_columns:
            raise SystemExit(
                f"Conditions CSV contains unknown columns: {unknown_columns}. "
                f"Accepted columns: {sorted(allowed_columns)}."
            )
    else:
        source = None
        conditions = pd.DataFrame([{**direct_conditions, "n_samples": args.n_samples}])
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    condition_rows: list[dict[str, object]] = []
    for row_index, row in conditions.iterrows():
        row_conditions = {
            str(field): value
            for field, value in row.items()
            if field not in {"n_samples", "phenotype", "body_site"}
            and not pd.isna(value)
        }
        if "phenotype" in row and not pd.isna(row["phenotype"]):
            row_conditions["Phenotype_Tokens"] = row["phenotype"]
        if "body_site" in row and not pd.isna(row["body_site"]):
            row_conditions["Body_Site_Coarse"] = row["body_site"]
        n_samples = _positive_sample_count(row.get("n_samples", args.n_samples))
        phenotype = str(row_conditions.pop("Phenotype_Tokens", args.phenotype))
        body_site = str(row_conditions.pop("Body_Site_Coarse", args.body_site))
        request: dict[str, object] = {
            "n_samples": n_samples,
            "conditions": row_conditions,
        }
        condition_fields = set(simulator.diffusion_config.primary_fields)
        if "Phenotype_Tokens" in condition_fields:
            request["phenotype"] = phenotype
        if "Body_Site_Coarse" in condition_fields:
            request["body_site"] = body_site
        condition_rows.append(request)

    model_name = None
    if selected_model is not None:
        model_name = (
            "custom"
            if selected_model == "custom"
            else official_model_name(selected_model, use_body_system)
        )
    if args.bank:
        source_config = {
            "generation_source": "bank",
            "bank": {
                "path": str(root),
                "type": str(simulator.manifest["bank_type"]),
                "schema_version": int(simulator.manifest.get("schema_version", 1)),
            },
        }
    else:
        source_config = {
            "generation_source": "model",
            "model": {
                "type": simulator.bundle.model_type,
                "name": model_name,
                "path": str(root),
                "condition_fields": list(simulator.diffusion_config.primary_fields),
                "supports_body_system": simulator.bundle.supports_body_system,
            },
            "device": args.device,
        }
    if source is not None:
        source_config["condition_table_source"] = str(source)
    result = _generate_from_condition_table(
        simulator,
        condition_rows,
        output_mode=args.output_mode,
        guidance_scale=args.guidance_scale,
        support_temperature=args.support_temperature,
        seed=args.seed,
        library_size=args.library_size,
        use_body_system=use_body_system,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_in_memory_mb=args.max_in_memory_mb,
        load=False,
        config=source_config,
    )
    return result.output_dir


def _run_benchmark(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Path | None:
    from .benchmark_config import (
        design_cell_counts,
        list_presets,
        load_custom_config,
        load_preset,
        resolve_preset,
        scenario_truth_applicability,
        write_config_template,
    )

    if args.list_presets:
        for name, description in list_presets():
            print(f"{name}\t{description}")
        return None
    if args.show_preset:
        print(
            yaml.safe_dump(
                load_preset(args.show_preset), sort_keys=False, allow_unicode=True
            ).rstrip()
        )
        return None
    if args.list_parameters:
        _print_benchmark_parameters(parser, args.preset)
        return None
    if args.write_config_template:
        print(write_config_template(args.write_config_template))
        return None
    if not args.preset and not args.config:
        raise SystemExit(
            "benchmark requires --preset NAME or --config PATH; use --list-presets to choose."
        )
    overrides = _benchmark_overrides(args)
    if args.config and overrides:
        options = ", ".join(
            f"--{name.replace('_', '-')}"
            for name in vars(args)
            if name in BENCHMARK_OVERRIDE_PATHS
        )
        raise SystemExit(
            f"--config is complete and cannot be combined with scientific CLI overrides: {options}."
        )
    scenario = (
        load_custom_config(args.config)
        if args.config
        else resolve_preset(args.preset, overrides)
    )
    if args.dry_run:
        resolved = scenario.as_dict()
        resolved["actual_cell_counts"] = design_cell_counts(scenario)
        resolved["truth_applicability"] = scenario_truth_applicability(scenario)
        print(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True).rstrip())
        return None
    if not args.output_dir:
        raise SystemExit("benchmark generation requires --output-dir.")

    from ._benchmark import generate_benchmark

    if args.bank:
        from .bank import load_bank

        simulator = load_bank(args.bank)
    else:
        from .pretrained import resolve_model_directory
        from .simulator import Simulator

        root = resolve_model_directory(
            scenario.model.type,
            scenario.model.directory,
            bool(scenario.model.use_body_system),
        )
        simulator = Simulator.from_directory(root, scenario.runtime.device)
    return generate_benchmark(
        simulator,
        scenario,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        max_in_memory_mb=args.max_in_memory_mb,
        load=False,
    ).output_dir


def _run_prepare(args: argparse.Namespace) -> Path:
    from .data import prepare_data

    return prepare_data(
        abundance_path=args.abundance,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        sample_column=args.sample_column,
        phenotype_column=args.phenotype_column,
        body_site_column=args.body_site_column,
        split_column=args.split_column,
        split_file=args.split_file,
        body_system_map=args.body_system_map,
        taxonomy_path=args.taxonomy,
        biological_fields=args.biological_fields
        or ("Phenotype_Tokens", "Body_Site_Coarse"),
        multilabel_fields=(
            args.multilabel_fields
            if args.multilabel_fields is not None
            else ("Phenotype_Tokens",)
        ),
        technical_fields=(
            args.technical_fields
            if args.technical_fields is not None
            else ("Instrument", "Variable_Region", "Sequencing_Type", "Library_Layout")
        ),
        seed=args.seed,
        overwrite=args.overwrite,
    )


def _run_train_tokenizer(args: argparse.Namespace) -> Path:
    from ._train_tokenizer import train_tokenizer

    config = args.config or str(
        files("mbbench.defaults").joinpath(f"tokenizer_{args.model}.yaml")
    )
    return train_tokenizer(
        config_path=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def _run_train_diffusion(args: argparse.Namespace) -> Path:
    from ._train_diffusion import train_diffusion

    config = args.config or str(files("mbbench.defaults").joinpath("diffusion.yaml"))
    return train_diffusion(
        config_path=config,
        data_dir=args.data_dir,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        output_dir=args.output_dir,
        use_body_system=args.use_body_system,
        device_name=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def _run_package_model(args: argparse.Namespace) -> Path:
    from ._package_model import package_model

    mapping = Path(args.data_dir) / "body_site_to_system.csv"
    return package_model(
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        supports_body_system=args.use_body_system,
        body_system_map=mapping if args.use_body_system else None,
        overwrite=args.overwrite,
    )


def _run_download_model(args: argparse.Namespace) -> Path:
    import sys

    from .pretrained import (
        OFFICIAL_MODEL_URLS,
        download_model,
        model_cache_dir,
        official_model_name,
        prompt_install_directory,
    )

    name = official_model_name(args.model, args.use_body_system)
    url = args.url or OFFICIAL_MODEL_URLS[name]
    if not url:
        raise SystemExit(
            "This release has no registered official archive yet; provide --url."
        )
    default_dir = model_cache_dir(name)
    if args.install_dir:
        target_dir = Path(args.install_dir).expanduser().resolve()
    elif sys.stdin.isatty() and sys.stdout.isatty():
        target_dir = prompt_install_directory(default_dir)
    else:
        target_dir = default_dir
    return download_model(url=url, target_dir=target_dir).root


def _run_list_datasets(args: argparse.Namespace) -> None:
    from .datasets import list_datasets

    entries = list_datasets(version=args.dataset_version)
    print("preset\tseed\tsamples\tbatches\tversion")
    for entry in entries:
        print(
            f"{entry['preset']}\t{entry['seed']}\t{entry.get('n_samples', '')}\t"
            f"{entry.get('n_batches', '')}\t{entry['version']}"
        )


def _run_show_dataset(args: argparse.Namespace) -> None:
    from .datasets import show_dataset

    print(
        yaml.safe_dump(
            show_dataset(args.preset, version=args.dataset_version),
            sort_keys=False,
            allow_unicode=True,
        ).rstrip()
    )


def _run_download_dataset(args: argparse.Namespace) -> Path:
    from .datasets import install_dataset

    root, _ = install_dataset(
        args.preset,
        seed=args.seed,
        version=args.dataset_version,
        install_dir=args.install_dir,
        overwrite=args.overwrite,
    )
    return root


def _run_import_dataset(args: argparse.Namespace) -> Path:
    from .datasets import install_dataset_archive

    return install_dataset_archive(
        args.archive,
        install_dir=args.install_dir,
        overwrite=args.overwrite,
    )


def _run_list(args: argparse.Namespace) -> None:
    if args.bank:
        from .bank import load_bank

        source = load_bank(args.bank)
        condition_fields = source.condition_fields
        condition_values = source.condition_values
    else:
        from .bundle import ModelBundle
        from .pretrained import resolve_model_directory

        selected_model = args.model or GENERATION_DEFAULTS["model"]
        root = resolve_model_directory(
            selected_model, args.model_dir, args.use_body_system
        )
        bundle = ModelBundle.load(root)
        condition_fields = tuple(bundle.config["condition_fields"])
        condition_values = bundle.condition_values
    fields = list(args.field or [])
    if args.phenotypes:
        fields.append("Phenotype_Tokens")
    if args.body_sites:
        fields.append("Body_Site_Coarse")
    if not fields:
        fields = list(condition_fields)
    fields = list(dict.fromkeys(fields))
    unknown_fields = sorted(set(fields) - set(condition_fields))
    if unknown_fields:
        raise SystemExit(f"Model does not use condition fields: {unknown_fields}")
    for field in fields:
        print(f"{field}:")
        for value in condition_values(field):
            print(f"  {value}")


def _run_evaluate(args: argparse.Namespace) -> Path:
    from .evaluate import evaluate_generation

    evaluate_generation(
        real_abundance=args.real_abundance,
        synthetic_abundance=args.synthetic_abundance,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    return Path(args.output_dir).resolve()


def _named_inputs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--input must use METHOD=CSV.")
        method, path = value.split("=", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise SystemExit("--input must use nonempty METHOD=CSV values.")
        if method in result:
            raise SystemExit(f"--input repeats method {method!r}.")
        result[method] = path
    return result


def _run_evaluate_batch(args: argparse.Namespace) -> Path:
    from .benchmark_evaluation import evaluate_batch

    result = evaluate_batch(
        args.benchmark,
        _named_inputs(args.inputs),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        prevalence_min=args.prevalence_min,
        maximum_relative_abundance_min=args.maximum_relative_abundance_min,
        calculate_scores=args.calculate_scores,
    )
    return result.output_dir


def _run_evaluate_da(args: argparse.Namespace) -> Path:
    from .benchmark_evaluation import evaluate_da, evaluate_da_many

    if args.manifest:
        if args.benchmark or args.inputs:
            raise SystemExit("--manifest cannot be combined with --benchmark or --input.")
        result = evaluate_da_many(
            args.manifest,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.summary_seed,
        )
    else:
        if not args.benchmark or not args.inputs:
            raise SystemExit(
                "Provide --manifest, or provide --benchmark with at least one --input."
            )
        result = evaluate_da(
            args.benchmark,
            _named_inputs(args.inputs),
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    return result.output_dir


def _run_summarize_batch(args: argparse.Namespace) -> Path:
    from .benchmark_evaluation import summarize_batch

    summarize_batch(
        args.input, output_dir=args.output_dir, overwrite=args.overwrite
    )
    return Path(args.output_dir).resolve()


def _run_summarize_da(args: argparse.Namespace) -> Path:
    from .benchmark_evaluation import summarize_da

    summarize_da(
        args.input,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    return Path(args.output_dir).resolve()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_model_arguments(args)
    if args.command == "prepare":
        output = _run_prepare(args)
    elif args.command == "train-tokenizer":
        output = _run_train_tokenizer(args)
    elif args.command == "train-diffusion":
        output = _run_train_diffusion(args)
    elif args.command == "package-model":
        output = _run_package_model(args)
    elif args.command == "download-model":
        output = _run_download_model(args)
    elif args.command == "list-datasets":
        _run_list_datasets(args)
        return
    elif args.command == "show-dataset":
        _run_show_dataset(args)
        return
    elif args.command == "download-dataset":
        output = _run_download_dataset(args)
    elif args.command == "import-dataset":
        output = _run_import_dataset(args)
    elif args.command == "list-conditions":
        _run_list(args)
        return
    elif args.command == "generate":
        output = _run_generate(args)
    elif args.command == "benchmark":
        output = _run_benchmark(args, parser)
        if output is None:
            return
    elif args.command == "evaluate-batch":
        output = _run_evaluate_batch(args)
    elif args.command == "evaluate-da":
        output = _run_evaluate_da(args)
    elif args.command == "summarize-batch":
        output = _run_summarize_batch(args)
    elif args.command == "summarize-da":
        output = _run_summarize_da(args)
    else:
        output = _run_evaluate(args)
    print(output)


if __name__ == "__main__":
    main()
