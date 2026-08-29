# ./run_experiments.py

from pathlib import Path
import time
import yaml

from src.utils.config import load_config
from src.training.trainer import run_training
from src.training.fine_tuner import run_fine_tuning


# Experimental matrix (3 architectures x 2 addresses = 6 configurations)
EXPERIMENTS = [
    # --------------------------------------------------------------------------
    # Aym -> Spa
    # --------------------------------------------------------------------------
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
    # --------------------------------------------------------------------------
    # Spa -> Aym
    # --------------------------------------------------------------------------
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


def run_orchestrator(base_config_path: str = "config.yaml") -> None:
    total_start_time = time.perf_counter()
    temp_configs_dir = Path("configs_generated")
    temp_configs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(base_config_path)
    total_exps = len(EXPERIMENTS)

    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENTAL BENCHMARK ({total_exps} CONFIGURATIONS)")
    print("Each setup will execute: [1] Base Biblical Training -> [2] Domain Fine-Tuning")
    print("=" * 80 + "\n")

    for idx, exp in enumerate(EXPERIMENTS, 1):
        exp_start_time = time.perf_counter()

        # Specific configuration for the run
        current_cfg = base_cfg.copy()
        current_cfg["src"] = exp["src"]
        current_cfg["tgt"] = exp["tgt"]
        current_cfg["encoder_layers"] = exp["encoder_layers"]
        current_cfg["decoder_layers"] = exp["decoder_layers"]

        tag = f"{exp['src']}_to_{exp['tgt']}_Enc{exp['encoder_layers']}_Dec{exp['decoder_layers']}"
        exp_config_file = temp_configs_dir / f"config_{tag}.yaml"

        with open(exp_config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(current_cfg, f, sort_keys=False)

        print("\n" + "#" * 80)
        print(f"EXPERIMENT {idx}/{total_exps}: {exp['description']}")
        print(f"Generated Config: {exp_config_file}")
        print("#" * 80)

        # Phase 1: Base Training in Biblical Corpus and Evaluation (ID and OOD)
        print(f"\n>>> [PHASE 1/2] Launching Base Training ({tag})...")
        run_training(config_path=exp_config_file)

        # Phase 2: Domain Fine-Tuning with Dialogues and Evaluation (ID and OOD)
        print(f"\n>>> [PHASE 2/2] Launching Domain Fine-Tuning ({tag})...")
        run_fine_tuning(config_path=exp_config_file)

        exp_elapsed = time.perf_counter() - exp_start_time
        print(f"\n✓ Completed Experiment {idx}/{total_exps} ({tag}) in {exp_elapsed / 3600:.2f} hours.")

    total_elapsed = time.perf_counter() - total_start_time
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS SUCCESSFULLY FINISHED! Total Duration: {total_elapsed / 3600:.2f} hours")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_orchestrator(base_config_path = "config.yaml")