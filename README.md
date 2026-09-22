# Aymara–Spanish Neural Machine Translation

Code, configurations, and experimental results for bidirectional Aymara–Spanish neural machine translation in a low-resource setting.

The study compares **encoder–decoder depth allocation under a fixed budget of eight Transformer layers**: 4E–4D, 6E–2D, and 2E–6D. The layer budget is fixed; the number of parameters is not identical because decoder layers contain additional cross-attention. This is a controlled architecture study, not a state-of-the-art benchmark.

Each architecture is trained in both directions on a biblical corpus and then fine-tuned on a conversational corpus. Both test domains are evaluated before and after fine-tuning.

## Experimental design

| Setting | Value |
|---|---|
| Directions | Aymara → Spanish (`aym_to_spa`), Spanish → Aymara (`spa_to_aym`) |
| Encoder / decoder layers | 4 / 4, 6 / 2, 2 / 6 |
| Training seeds | `111`, `222`, `333` |
| Dataset split seed | `1989`, held fixed across all training seeds |
| Runs | 18 base-training runs and 18 subsequent fine-tuning runs |
| Tokenizer | Shared SentencePiece BPE, 8,000 pieces; trained on both languages of the biblical training split only |
| Model width / attention heads / FFN width | 256 / 8 / 1,024 |
| Dropout | 0.2 |
| Base training | 150 epochs, CUDA batch size 64, peak learning rate 0.0005 |
| Fine-tuning | 30 epochs, batch size 16, learning rate 0.0001 |
| Checkpoint selection | Highest validation chrF++ at the evaluated epochs |
| Validation decoding | Greedy decoding; full validation set every 10 base epochs / 2 fine-tuning epochs, and at the last epoch |
| Test decoding | Beam size 4, maximum length 150, length-penalty alpha 0.6 |

The training code uses AdamW, label smoothing, gradient clipping, and CUDA mixed precision. See [`config.yaml`](config.yaml), [`src/training/trainer.py`](src/training/trainer.py), and [`src/training/fine_tuner.py`](src/training/fine_tuner.py) for the full settings. Each result JSON stores its own configuration and selected checkpoint epoch.

### Dataset sizes

| Corpus | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Biblical | 108,823 | 6,048 | 6,033 | 120,904 |
| Conversational | 1,132 | 242 | 244 | 1,618 |

These are parallel-pair counts recorded in the project analysis. Biblical splitting groups rows by USFM verse identifier, keeping edition variants of a verse in the same split. Conversational splitting shuffles parallel pairs with the fixed split seed.

## Repository layout

| Path | Purpose |
|---|---|
| `1_scrap_bibles.py` … `10_encode_book.py` | Extraction, alignment, preprocessing, splitting, tokenizer training, and encoding |
| `11_run_transformer_experiments.py` | Multi-seed experiment orchestrator |
| `config.yaml` | Experiment and data-processing settings |
| `environment.yml` | Conda environment specification; uses `requirements.txt` |
| `requirements.txt` | Pinned direct dependencies |
| `src/data/` | Corpus processing and PyTorch datasets |
| `src/models/` | Transformer model |
| `src/tokenization/` | SentencePiece training and encoding |
| `src/training/` | Base training and fine-tuning |
| `src/utils/` | Configuration, metrics, parameter counts, reproducibility, and results analysis |
| `configs_generated/` | 18 generated YAML configurations for the reported runs |
| `results/` | 36 test-metric JSON files and 36 training-history CSV files |
| `paper_analysis.ipynb` | Corpus statistics, parameter counts, result aggregation, tables, and plots |
| `plots/` | Analysis figures, including `architecture_chrf_2x2.png` and `.pdf` |
| `tokenizer/` | `SentencePiece.model` and `SentencePiece.vocab` |
| `data/raw/book/book.pdf` | Conversational source work, under its own license |
| `data/` (other corpus files) | Local source/processed data; excluded from Git |
| `models/` | Local trained checkpoints; excluded from Git |

## Software and hardware

The three-seed experiments were run locally with the following environment, installed before the runs and unchanged afterward:

| Component | Experimental environment |
|---|---|
| Computer | ASUS TUF Gaming F15 FX506HF |
| Operating system | Windows 11 Home, build 26200, x64 |
| System RAM | Approximately 16 GB |
| GPU | NVIDIA GeForce RTX 2050, 4 GB VRAM |
| NVIDIA driver | 531.14 |
| Conda | 25.7.0 |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1; CUDA available |
| NumPy / pandas | 2.2.6 / 2.3.3 |
| SentencePiece / SacreBLEU | 0.2.2 / 2.6.0 |

The dependency files pin the project's direct dependencies rather than every package in `pip freeze`. They are not a complete lock of transitive packages or Conda build strings.

### Create a new environment

Run from the directory containing both `environment.yml` and `requirements.txt`:

```bash
conda env create -f environment.yml
conda activate nmt
python -m pip check
```

If `nmt` already exists, create a separate environment for verification:

```bash
conda env create -n nmt-repro -f environment.yml
conda activate nmt-repro
```

Alternatively, without the YAML file:

```bash
conda create -n nmt-repro python=3.10.20 pip
conda activate nmt-repro
python -m pip install -r requirements.txt
python -m pip check
```

The supplied requirements target the CUDA 12.1 PyTorch build on compatible Windows/Linux systems. They use the [official PyTorch CUDA 12.1 wheel index](https://download.pytorch.org/whl/cu121); see [PyTorch's previous-version installation instructions](https://pytorch.org/get-started/previous-versions/). Other platforms or CPU-only installations require a corresponding PyTorch build and are not the reported experimental environment.

Verify the active interpreter and GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
```

For notebooks, select this environment in a notebook frontend such as VS Code. A kernel can be registered with:

```bash
python -m ipykernel install --user --name nmt --display-name "Python (nmt)"
```

`ipykernel` is included; a standalone Jupyter Notebook/Lab frontend is not installed by these files. When using `nmt-repro`, use that name for the kernel instead.

## Data and licensing

### Biblical corpus

The project configuration identifies these editions:

| Language | Bible ID | Edition | Rights notice |
|---|---:|---|---|
| Aymara | 293 | Qullan Arunaca | Sociedad Bíblica Boliviana |
| Aymara | 2250 | Qullan Arunaka DC | Sociedad Bíblica Boliviana |
| Spanish | 4278 | Dios Habla Hoy, 4th edition | Sociedades Bíblicas Unidas |
| Spanish | 146 | Reina Valera Contemporánea | Sociedades Bíblicas Unidas |

The repository does not redistribute these source texts or their aligned, cleaned, split, or tokenized corpora. Tokenized TSV files also retain text columns. Obtain the source editions independently and comply with their applicable permissions and terms; the availability of extraction code is not a grant of rights to the content.

### Conversational corpus

The source is **AYMARA ARUSKIPAWINAKA: Conversaciones en aimara**, edited by **Román Pairumani Ajacopa** and **Alejandra Bertha Carrasco Lima**, first electronic edition, La Paz, Bolivia, January 2022. The credits on page 4 state **Creative Commons Attribution-NonCommercial 4.0 International**.

The source PDF is retained at [`data/raw/book/book.pdf`](data/raw/book/book.pdf), with its original credits and license notice. See the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/). Retain attribution, link to the license, indicate changes when making adaptations, and observe the noncommercial restriction. The project extracts and normalizes text locally; generated conversational TSV files are excluded from Git as reproducible intermediate artifacts.

### Publication policy

The repository tracks code, configurations, aggregate metrics, training histories, figures, and the SentencePiece tokenizer files. Checkpoints and corpora are excluded. Tokenizer files are retained as reproducibility artifacts; their inclusion does not grant rights to the underlying source works.

Do not commit corpus text in notebook outputs, exported previews, or prediction/reference dumps. The publication copy of `paper_analysis.ipynb` omits its two saved corpus-preview outputs while retaining its code and numerical analyses. Running those preview cells again will recreate the outputs locally; clear them before committing.

`.gitignore` only prevents new files from being tracked. Previously tracked files need to be removed from the Git index separately, and previous commits remain in repository history.

## Reproduce the pipeline

Run all commands from the project root. Source access is required for the full data/training workflow; cloning alone does not provide the biblical corpus.

1. Obtain authorized access to the biblical editions. `1_scrap_bibles.py` writes JSON source files under `data/raw/bible/aym/` (`293.json`, `2250.json`) and `data/raw/bible/spa/` (`146.json`, `4278.json`). If these files already exist in the expected format, skip extraction.
2. Ensure `data/raw/book/book.pdf` is present.
3. Run the applicable preparation steps in this order:

```bash
python 1_scrap_bibles.py
python 2_scrap_book.py
python 3_align_verses.py
python 4_preprocess_bible.py
python 5_preprocess_book.py
python 6_split_bible.py
python 7_split_book.py
```

The scraper depends on an external website whose availability and layout can change. Previously obtained authorized source files can be processed without downloading them again.

For reproduction using the supplied tokenizer, keep `tokenizer/SentencePiece.model` and `.vocab`. To train a new tokenizer from the biblical training split, run:

```bash
python 8_train_tokenizer.py
```

This replaces the tokenizer files. Do not mix tokenized datasets or checkpoints built with different tokenizers. Encode both corpora with the selected tokenizer:

```bash
python 9_encode_bible.py
python 10_encode_book.py
```

### Run the reported experiments

```bash
python 11_run_transformer_experiments.py --seeds 111 222 333
```

The orchestrator runs six conditions for each seed sequentially. Each condition performs base training, evaluates both test sets, fine-tunes from the selected base checkpoint, and evaluates both test sets again. It generates seed-specific configurations, checkpoints, metrics, and histories.

To execute only one seed, or supply a different base configuration:

```bash
python 11_run_transformer_experiments.py --seeds 111
python 11_run_transformer_experiments.py --seeds 111 222 333 --config config.yaml
```

Repeated runs with the same tags write to the same output paths. Back up results before rerunning; the orchestrator does not implement automatic resume or skip completed runs. Changing batch sizes or other settings creates a different experimental condition.

## Evaluation and analysis

SacreBLEU computes corpus-level **chrF++** and **BLEU** on decoded hypotheses and SentencePiece-decoded references. The saved results use these signatures:

```text
chrF++: nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
BLEU:   nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
```

Each of the 36 metric JSON files contains results for both test domains, yielding 72 evaluation records. Aggregating over the three seeds produces 24 direction/architecture/stage/domain groups. Standard deviations below are sample standard deviations (`ddof=1`), not confidence intervals.

The following command summarizes the published JSON results without needing corpus files or model checkpoints:

```bash
python -c "from src.utils.results_analysis import load_test_results, summarize_test_results; print(summarize_test_results(load_test_results('results')).to_string(index=False))"
```

To regenerate the architecture figure:

```bash
python -c "from src.utils.results_analysis import load_test_results, summarize_test_results, plot_architecture_chrf; plot_architecture_chrf(summarize_test_results(load_test_results('results')), output_dir='plots')"
```

The full notebook also reads local corpus files for statistics and the tokenizer for parameter counts. Those sections cannot run from the public results alone.

## Results

Mean ± standard deviation across seeds 111, 222, and 333, recomputed from the 36 saved JSON files. `aym` denotes Aymara and `spa` Spanish. Each metric is reported on a 0–100 scale.

### Base model

| Direction | Architecture | Biblical chrF++ | Biblical BLEU | Conversational chrF++ | Conversational BLEU |
|---|---|---:|---:|---:|---:|
| aym → spa | 4E–4D | 42.96 ± 0.04 | 22.31 ± 0.06 | 23.39 ± 0.42 | 3.04 ± 0.40 |
| aym → spa | 6E–2D | 42.07 ± 0.13 | 21.43 ± 0.07 | 22.74 ± 0.32 | 3.72 ± 0.09 |
| aym → spa | 2E–6D | 42.72 ± 0.11 | 22.02 ± 0.06 | 23.53 ± 0.51 | 3.22 ± 0.18 |
| spa → aym | 4E–4D | 35.97 ± 0.01 | 7.44 ± 0.02 | 24.46 ± 0.59 | 0.72 ± 0.33 |
| spa → aym | 6E–2D | 34.95 ± 0.04 | 6.80 ± 0.09 | 23.83 ± 0.13 | 1.05 ± 0.14 |
| spa → aym | 2E–6D | 35.77 ± 0.08 | 7.27 ± 0.06 | 24.47 ± 0.09 | 0.84 ± 0.38 |

### Fine-tuned model

| Direction | Architecture | Biblical chrF++ | Biblical BLEU | Conversational chrF++ | Conversational BLEU |
|---|---|---:|---:|---:|---:|
| aym → spa | 4E–4D | 24.24 ± 0.30 | 6.69 ± 0.19 | 59.35 ± 1.53 | 38.20 ± 2.32 |
| aym → spa | 6E–2D | 23.57 ± 2.48 | 6.60 ± 1.93 | 57.86 ± 1.46 | 37.02 ± 1.53 |
| aym → spa | 2E–6D | 24.47 ± 0.47 | 6.73 ± 0.30 | 59.46 ± 0.52 | 37.37 ± 0.66 |
| spa → aym | 4E–4D | 18.97 ± 0.05 | 0.85 ± 0.03 | 58.91 ± 0.76 | 31.49 ± 2.81 |
| spa → aym | 6E–2D | 17.89 ± 0.52 | 0.70 ± 0.06 | 56.94 ± 0.36 | 28.54 ± 0.61 |
| spa → aym | 2E–6D | 19.62 ± 0.15 | 0.92 ± 0.06 | 57.81 ± 0.56 | 30.90 ± 1.16 |


Fine-tuning improves conversational scores while biblical scores decline in these runs. These observations concern the evaluated corpora and training protocol; they do not establish general superiority or isolate a causal mechanism. Small differences between architectures should be interpreted alongside the variation across seeds.

## Reproducibility limits

The code seeds Python, NumPy, and PyTorch, seeds data-loader workers, and configures cuDNN for deterministic behavior. Exact numerical equality across hardware, library builds, or regenerated data/tokenizers is not guaranteed. The published JSON files and their embedded configurations are the record of the reported runs.

The environment specifications were assembled from the author's reported versions and the current source imports. They have not been validated by rerunning all experiments or by a fresh Windows GPU installation during this documentation update.

## License and citation

Original project code is covered by the [MIT License](LICENSE), copyright 2026 Josiel Corbera. Third-party source works are governed by their own terms; the software license does not relicense them.

The accompanying manuscript is in preparation. Until final publication metadata are available, cite this repository and the specific commit used: [wjosielct/aymara-spanish-nmt](https://github.com/wjosielct/aymara-spanish-nmt). Also acknowledge the original data sources where applicable.
