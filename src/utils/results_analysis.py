# ==============================
# ./src/utils/results_analysis.py
# ==============================

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_test_results(
        results_dir: str | Path = "results"
) -> pd.DataFrame:
    """
    Loads all base-training and fine-tuning test JSON files.

    Returns one row per:
    seed x direction x architecture x stage x domain
    """

    results_dir = Path(results_dir)

    json_files = sorted(
        list(results_dir.glob("trained_test_metrics_*.json"))
        + list(results_dir.glob("finetuned_test_metrics_*.json"))
    )

    if not json_files:
        raise FileNotFoundError(f"No test metric JSON files were found in: {results_dir}")

    rows = []

    for json_path in json_files:

        with open(json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        cfg = payload["config"]

        seed = cfg["seed"]
        src = cfg["src"]
        tgt = cfg["tgt"]
        enc = cfg["encoder_layers"]
        dec = cfg["decoder_layers"]

        direction = f"{src}_to_{tgt}"
        architecture = f"{enc}E-{dec}D"

        experiment_type = payload["experiment_type"]

        if experiment_type == "Base-Training":
            stage = "Base"
        elif experiment_type == "Fine-Tuning":
            stage = "Fine-tuned"
        else:
            raise ValueError(
                f"Unknown experiment_type '{experiment_type}' "
                f"in file: {json_path.name}"
            )

        domain_map = {
            "biblical_results": "Biblical",
            "conversational_results": "Conversational"
        }

        for json_key, domain in domain_map.items():

            metrics = payload[json_key]

            rows.append({
                "seed": seed,
                "src": src,
                "tgt": tgt,
                "direction": direction,
                "encoder_layers": enc,
                "decoder_layers": dec,
                "architecture": architecture,
                "stage": stage,
                "domain": domain,
                "test_pairs": metrics["test_pairs"],
                "BLEU": metrics["bleu_score"],
                "chrF++": metrics["chrf_score"],
                "best_checkpoint_epoch": payload["best_checkpoint_epoch"],
                "total_execution_time": payload["total_execution_time"],
                "file": json_path.name
            })

    df = pd.DataFrame(rows)

    return df



def summarize_test_results(
        results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes mean and sample standard deviation across seeds.
    """

    summary_df = (
        results_df
        .groupby(
            [
                "direction",
                "architecture",
                "stage",
                "domain"
            ],
            as_index=False
        )
        .agg(
            n_seeds=("seed", "nunique"),
            BLEU_mean=("BLEU", "mean"),
            BLEU_sd=("BLEU", "std"),
            chrF_mean=("chrF++", "mean"),
            chrF_sd=("chrF++", "std")
        )
    )

    return summary_df



def generate_latex_table(summary_df) -> str:
    """
    Generates a LaTeX table with mean ± standard deviation
    for BLEU and chrF++ across multiple seeds.

    Expected columns in summary_df:
        direction
        architecture
        stage
        domain
        BLEU_mean
        BLEU_sd
        chrF_mean
        chrF_sd

    Returns:
        str: Complete LaTeX table.
    """

    direction_order = [
        "aym_to_spa",
        "spa_to_aym"
    ]

    architecture_order = [
        "4E-4D",
        "6E-2D",
        "2E-6D"
    ]

    stage_order = [
        "Base",
        "Fine-tuned"
    ]

    domain_order = [
        "Biblical",
        "Conversational"
    ]

    direction_labels = {
        "aym_to_spa": r"$\text{Aym} \to \text{Spa}$",
        "spa_to_aym": r"$\text{Spa} \to \text{Aym}$"
    }

    architecture_labels = {
        "4E-4D": r"$4\text{E}\text{--}4\text{D}$",
        "6E-2D": r"$6\text{E}\text{--}2\text{D}$",
        "2E-6D": r"$2\text{E}\text{--}6\text{D}$"
    }

    def format_cell(mean, sd):
        """
        Formats a metric as mean ± SD with two decimals.
        """
        return rf"{mean:.2f}$\pm${sd:.2f}"

    def get_metrics(
        direction,
        architecture,
        stage,
        domain
    ):
        row = summary_df[
            (summary_df["direction"] == direction)
            & (summary_df["architecture"] == architecture)
            & (summary_df["stage"] == stage)
            & (summary_df["domain"] == domain)
        ]

        if len(row) != 1:
            raise ValueError(
                "Expected exactly one row for "
                f"{direction}, {architecture}, "
                f"{stage}, {domain}, "
                f"but found {len(row)}."
            )

        row = row.iloc[0]

        bleu = format_cell(
            row["BLEU_mean"],
            row["BLEU_sd"]
        )

        chrf = format_cell(
            row["chrF_mean"],
            row["chrF_sd"]
        )

        return bleu, chrf

    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")

    lines.append(
        r"    \caption{Translation performance for different "
        r"encoder--decoder depth allocations on biblical and "
        r"conversational test sets before and after fine-tuning. "
        r"Results are reported as mean $\pm$ standard deviation "
        r"over three random seeds.}"
    )

    lines.append(
        r"    \label{tab:nmt_experimental_results}"
    )

    lines.append(r"    \scriptsize")
    lines.append(r"    \setlength{\tabcolsep}{3.2pt}")
    lines.append(r"    \renewcommand{\arraystretch}{1.15}")

    lines.append(
        r"    \begin{tabular}{llcccccccc}"
    )

    lines.append(r"        \hline")

    lines.append(
        r"        \multirow{3}{*}{\textbf{Direction}} & "
        r"\multirow{3}{*}{\textbf{Architecture}} & "
        r"\multicolumn{4}{c}{\textbf{Base Model}} & "
        r"\multicolumn{4}{c}{\textbf{Fine-Tuned Model}} \\"
    )

    lines.append(r"        \cline{3-10}")

    lines.append(
        r"        & & "
        r"\multicolumn{2}{c}{\textbf{Biblical}} & "
        r"\multicolumn{2}{c}{\textbf{Conversational}} & "
        r"\multicolumn{2}{c}{\textbf{Biblical}} & "
        r"\multicolumn{2}{c}{\textbf{Conversational}} \\"
    )

    lines.append(r"        \cline{3-10}")

    lines.append(
        r"        & & "
        r"\textbf{BLEU} & \textbf{chrF++} & "
        r"\textbf{BLEU} & \textbf{chrF++} & "
        r"\textbf{BLEU} & \textbf{chrF++} & "
        r"\textbf{BLEU} & \textbf{chrF++} \\"
    )

    lines.append(r"        \hline")

    for direction in direction_order:

        for arch_idx, architecture in enumerate(
            architecture_order
        ):

            row_values = []

            for stage in stage_order:
                for domain in domain_order:

                    bleu, chrf = get_metrics(
                        direction=direction,
                        architecture=architecture,
                        stage=stage,
                        domain=domain
                    )

                    row_values.extend([
                        bleu,
                        chrf
                    ])

            if arch_idx == 0:
                direction_cell = (
                    rf"\multirow{{3}}{{*}}"
                    rf"{{{direction_labels[direction]}}}"
                )
            else:
                direction_cell = ""

            row = (
                f"        {direction_cell} & "
                f"{architecture_labels[architecture]} & "
                + " & ".join(row_values)
                + r" \\"
            )

            lines.append(row)

        lines.append(r"        \hline")

    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def plot_architecture_chrf(
    summary_df: pd.DataFrame,
    output_dir: str | Path = "plots",
    filename: str = "architecture_chrf_2x2",
    expected_seeds: int = 3,
):
    """
    Plots mean chrF++ ± sample SD across training seeds.

    Rows:
        Base model evaluated on biblical text.
        Adapted model evaluated on conversational text.

    Columns:
        Aymara -> Spanish.
        Spanish -> Aymara.

    Vertical scales are shared within each row.
    Saves a vector PDF and a 300-dpi PNG.
    Returns (fig, axes).
    """

    required_columns = {
        "direction", "architecture", "stage", "domain",
        "n_seeds", "chrF_mean", "chrF_sd",
    }
    missing_columns = required_columns - set(summary_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    architectures = ["4E-4D", "6E-2D", "2E-6D"]
    architecture_labels = ["4E–4D", "6E–2D", "2E–6D"]

    directions = [
        ("aym_to_spa", "Aymara → Spanish"),
        ("spa_to_aym", "Spanish → Aymara"),
    ]

    # Stage names match summary_df; display labels are independent.
    conditions = [
        ("Base", "Biblical", "Base / Biblical", "#0072B2"),
        (
            "Fine-tuned",
            "Conversational",
            "Adapted / Conversational",
            "#D55E00",
        ),
    ]

    # Validate the required data before creating the figure.
    panels = {}

    for row_idx, (stage, domain, _, _) in enumerate(conditions):
        for col_idx, (direction, _) in enumerate(directions):
            panel = summary_df.loc[
                (summary_df["stage"] == stage)
                & (summary_df["domain"] == domain)
                & (summary_df["direction"] == direction)
                & summary_df["architecture"].isin(architectures)
            ].copy()

            if (
                len(panel) != len(architectures)
                or panel["architecture"].duplicated().any()
                or set(panel["architecture"]) != set(architectures)
            ):
                raise ValueError(
                    "Expected one summary row per architecture for "
                    f"{direction}, {stage}, {domain}."
                )

            panel = (
                panel.set_index("architecture")
                .loc[architectures]
            )

            if not panel["n_seeds"].eq(expected_seeds).all():
                raise ValueError(
                    f"Expected {expected_seeds} seeds per architecture "
                    f"for {direction}, {stage}, {domain}."
                )

            means = panel["chrF_mean"].to_numpy(dtype=float)
            sds = panel["chrF_sd"].to_numpy(dtype=float)

            if (
                not np.isfinite(means).all()
                or not np.isfinite(sds).all()
                or (sds < 0).any()
            ):
                raise ValueError(
                    f"Invalid means or SDs for {direction}, "
                    f"{stage}, {domain}."
                )

            panels[row_idx, col_idx] = (means, sds)

    style = {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.2, 4.4),
            sharey="row",
            layout="constrained",
        )

        x = np.arange(len(architectures))

        for row_idx, (_, _, condition_label, color) in enumerate(
            conditions
        ):
            for col_idx, (_, direction_label) in enumerate(directions):
                ax = axes[row_idx, col_idx]
                means, sds = panels[row_idx, col_idx]

                ax.errorbar(
                    x,
                    means,
                    yerr=sds,
                    fmt="o",
                    linestyle="none",
                    color=color,
                    ecolor=color,
                    markersize=6,
                    elinewidth=1.3,
                    capsize=4,
                    capthick=1.3,
                    zorder=3,
                )

                letter = "abcd"[row_idx * 2 + col_idx]
                ax.set_title(f"({letter}) {direction_label}")
                ax.set_xticks(x, architecture_labels)
                ax.set_xlim(-0.4, len(architectures) - 0.6)

                ax.set_axisbelow(True)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

                if row_idx == 1:
                    ax.set_xlabel("Architecture")

            axes[row_idx, 0].set_ylabel(
                f"{condition_label}\nchrF++"
            )

            # Common limits within each row, including error bars.
            lower = min(
                np.min(
                    panels[row_idx, col][0]
                    - panels[row_idx, col][1]
                )
                for col in range(2)
            )
            upper = max(
                np.max(
                    panels[row_idx, col][0]
                    + panels[row_idx, col][1]
                )
                for col in range(2)
            )

            padding = max(0.4, 0.15 * (upper - lower))
            axes[row_idx, 0].set_ylim(
                lower - padding,
                upper + padding,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_dir / f"{filename}.pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"{filename}.png",
            dpi=300,
            bbox_inches="tight",
        )

    return fig, axes