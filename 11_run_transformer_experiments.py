# ====================
# ./run_transformer_experiments.py
# ====================

import argparse
from pathlib import Path
import time
import yaml

from src.utils.config import load_config
from src.training.trainer import run_training
from src.training.fine_tuner import run_fine_tuning


# Experimental matrix: 3 architectures x 2 translation directions
EXPERIMENTS = [
    # ----------------
    # Aym -> Spa
    # ----------------
    
    {
        "src": "aym",
        "tgt": "spa",
        "encoder_layers": 4,
        "decoder_layers": 4,
        "description": "Aymara to Spanish [Symmetric 4-4]"
    },
    {
        "src": "aym",
        "tgt": "spa",
        "encoder_layers": 6,
        "decoder_layers": 2,
        "description": "Aymara to Spanish [Asymmetric Deep-Encoder 6-2]"
    },
    {
        "src": "aym",
        "tgt": "spa",
        "encoder_layers": 2,
        "decoder_layers": 6,
        "description": "Aymara to Spanish [Asymmetric Deep-Decoder 2-6]"
    },

    # ------------------
    # Spa -> Aym
    # ------------------

    {
        "src": "spa",
        "tgt": "aym",
        "encoder_layers": 4,
        "decoder_layers": 4,
        "description": "Spanish to Aymara [Symmetric 4-4]"
    },
    {
        "src": "spa",
        "tgt": "aym",
        "encoder_layers": 6,
        "decoder_layers": 2,
        "description": "Spanish to Aymara [Asymmetric Deep-Encoder 6-2]"
    },
    {
        "src": "spa",
        "tgt": "aym",
        "encoder_layers": 2,
        "decoder_layers": 6,
        "description": "Spanish to Aymara [Asymmetric Deep-Decoder 2-6]"
    }
]


def run_orchestrator(
        seeds: list[int],
        base_config_path: str | Path = "config.yaml"
) -> None:

    total_start_time = time.perf_counter()

    exp_configs_dir = Path("configs_generated")
    exp_configs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(base_config_path)

    total_experiments = len(EXPERIMENTS) * len(seeds)
    current_experiment = 0

    print("\n" + "=" * 90)
    print("STARTING MULTI-SEED EXPERIMENTAL BENCHMARK")
    print("=" * 90)
    print(f"Seeds               : {seeds}")
    print(f"Seeds count         : {len(seeds)}")
    print(f"Configurations/seed : {len(EXPERIMENTS)}")
    print(f"Total experiments   : {total_experiments}")
    print(
        "Each experiment     : "
        "[1] Base Biblical Training -> [2] Domain Fine-Tuning"
    )
    print("=" * 90 + "\n")

    for seed_idx, seed in enumerate(seeds, start=1):

        print("\n" + "=" * 90)
        print(
            f"STARTING SEED {seed_idx}/{len(seeds)} "
            f"-> {seed}"
        )
        print("=" * 90)

        for exp in EXPERIMENTS:

            current_experiment += 1
            exp_start_time = time.perf_counter()

            # ------------------------------------------------------------------
            # Create experiment-specific configuration
            # ------------------------------------------------------------------
            current_cfg = base_cfg.copy()

            current_cfg["seed"] = seed
            current_cfg["src"] = exp["src"]
            current_cfg["tgt"] = exp["tgt"]
            current_cfg["encoder_layers"] = exp["encoder_layers"]
            current_cfg["decoder_layers"] = exp["decoder_layers"]

            tag = (
                f"{exp['src']}_to_{exp['tgt']}"
                f"_Enc{exp['encoder_layers']}"
                f"_Dec{exp['decoder_layers']}"
                f"_Seed{seed}"
            )

            exp_config_file = (
                exp_configs_dir / f"config_{tag}.yaml"
            )

            with open(
                exp_config_file,
                "w",
                encoding="utf-8"
            ) as file:
                yaml.safe_dump(
                    current_cfg,
                    file,
                    sort_keys=False
                )

            # ------------------------------------------------------------------
            # Experiment header
            # ------------------------------------------------------------------
            print("\n" + "#" * 90)
            print(
                f"EXPERIMENT "
                f"{current_experiment}/{total_experiments}"
            )
            print(
                f"Seed         : {seed}"
            )
            print(
                f"Configuration: {exp['description']}"
            )
            print(
                f"Generated CFG: {exp_config_file}"
            )
            print("#" * 90)

            # ------------------------------------------------------------------
            # Phase 1: Base training on Biblical Data
            # ------------------------------------------------------------------
            print(
                f"\n>>> [PHASE 1/2] "
                f"Launching Base Training ({tag})..."
            )

            run_training(config_path=exp_config_file)

            # ------------------------------------------------------------------
            # Phase 2: Domain fine-tuning on Conversational Data
            # ------------------------------------------------------------------
            print(
                f"\n>>> [PHASE 2/2] "
                f"Launching Domain Fine-Tuning ({tag})..."
            )

            run_fine_tuning(config_path=exp_config_file)

            # ------------------------------------------------------------------
            # Experiment timing
            # ------------------------------------------------------------------
            exp_elapsed = time.perf_counter() - exp_start_time

            print(
                f"\n✓ Completed Experiment "
                f"{current_experiment}/{total_experiments}"
            )
            print(
                f"  Tag     : {tag}"
            )
            print(
                f"  Duration: "
                f"{exp_elapsed / 3600:.2f} hours"
            )

        print("\n" + "=" * 90)
        print(
            f"SEED {seed} COMPLETED SUCCESSFULLY"
        )
        print("=" * 90 + "\n")

    total_elapsed = time.perf_counter() - total_start_time

    print("\n" + "=" * 90)
    print("ALL EXPERIMENTS SUCCESSFULLY FINISHED!")
    print(f"Seeds executed    : {seeds}")
    print(f"Total experiments : {total_experiments}")
    print(
        f"Total duration    : "
        f"{total_elapsed / 3600:.2f} hours"
    )
    print("=" * 90 + "\n")


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Run the Aymara-Spanish NMT experimental matrix "
            "for one or more training seeds."
        )
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Training seeds to execute. "
            "Example: --seeds 111 222 333"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help=(
            "Path to the base configuration file. "
            "Default: config.yaml"
        )
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    run_orchestrator(seeds=args.seeds, base_config_path=args.config)