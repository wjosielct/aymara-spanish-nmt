# Aymara–Spanish Neural Machine Translation

This repository contains the source code, experimental configurations, and selected experimental results for the study of neural machine translation between **Aymara and Spanish** in a low-resource setting.

The study investigates the effect of **encoder–decoder depth allocation** and **domain adaptation** on bidirectional Aymara–Spanish translation using a compact Transformer architecture.

## Repository contents

The repository contains the code required to reproduce the data-processing pipeline, tokenizer training, model training, domain fine-tuning, and evaluation procedures described in the accompanying study.

```text
aymara-spanish-nmt-code/
│
├── 1_scrap_bibles.py
├── 2_scrap_book.py
├── 3_align_verses.py
├── 4_preprocess_bible.py
├── 5_preprocess_book.py
├── 6_split_bible.py
├── 7_split_book.py
├── 8_train_tokenizer.py
├── 9_encode_bible.py
├── 10_encode_book.py
├── 11_run_experiments.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── tokenization/
│   ├── training/
│   └── utils/
│
├── configs_generated/
├── config.yaml
├── requirements.txt
├── analysis.ipynb
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── interim/
│   ├── splits/
│   └── tokenized/
│
├── models/
├── results/
├── plots/
└── tokenizer/
    ├── SentencePiece.model
    └── SentencePiece.vocab
```

The `results/` directory contains the metrics and training histories obtained from the experiments. The `plots/` directory contains figures generated for the analysis.

Due to copyright and redistribution restrictions, the biblical source texts and datasets derived from them are not included in the repository. The conversational source PDF is included under the license specified by the original work.

---

## Software requirements

The experiments were developed using **Python 3.10**.

The recommended Python version is:

```text
Python 3.10
```

The Python dependencies are listed in:

```text
requirements.txt
```

### Creating the environment

Using Conda or Miniconda:

```bash
conda create -n nmt python=3.10
conda activate nmt
```

Then install the required Python packages:

```bash
pip install -r requirements.txt
```

To verify the Python version:

```bash
python --version
```

It should report Python 3.10.x.

### GPU support

The original model training and experimental benchmark were executed on a **JupyterHub cluster at the Universidad Nacional de Ingeniería (UNI), Peru**, using an NVIDIA RTX A4000 GPU with 16 GB of VRAM.

The complete experimental benchmark consists of six configurations:

* Aymara → Spanish: 4 encoder / 4 decoder layers
* Aymara → Spanish: 6 encoder / 2 decoder layers
* Aymara → Spanish: 2 encoder / 6 decoder layers
* Spanish → Aymara: 4 encoder / 4 decoder layers
* Spanish → Aymara: 6 encoder / 2 decoder layers
* Spanish → Aymara: 2 encoder / 6 decoder layers

Each configuration includes base training on the biblical corpus followed by domain fine-tuning on the conversational corpus.

The code also supports execution on a local computer. However, reproducing the complete training benchmark may require a CUDA-compatible NVIDIA GPU and sufficient system memory. CPU execution may be possible for individual preprocessing and analysis steps, but full model training can be considerably slower.

---

# Data and licensing

## Biblical corpus

The biblical corpus used in this study was constructed from multiple Aymara and Spanish Bible editions.

The editions used were:

| Language | Bible code | Edition                     | Copyright                    |
| -------- | ---------: | --------------------------- | ---------------------------- |
| Aymara   |        293 | Qullan Arunaca              | © Sociedad Bíblica Boliviana |
| Aymara   |       2250 | Qullan Arunaka DC           | © Sociedad Bíblica Boliviana |
| Spanish  |       4278 | Dios Habla Hoy, 4th edition | © Sociedades Bíblicas Unidas |
| Spanish  |        146 | Reina Valera Contemporánea  | © Sociedades Bíblicas Unidas |

The original biblical texts and datasets derived from these texts are **not distributed in this repository** because of copyright and redistribution restrictions.

The following data are therefore excluded from the public repository:

```text
data/raw/bible/
data/interim/
data/clean/verses_aligned_clean.tsv
data/splits/*_bible.tsv
data/tokenized/*_bible_tokenized.tsv
```

The repository nevertheless provides the code used to process and prepare the biblical data.

The biblical data-processing pipeline includes:

1. Extraction of Bible verses.
2. Alignment of corresponding verses across Bible editions.
3. Text normalization and cleaning.
4. Sentence segmentation and filtering.
5. Dataset partitioning.
6. SentencePiece tokenizer training.
7. Encoding of the parallel data.

Because the copyrighted biblical texts are not redistributed, users who wish to reproduce the biblical corpus must obtain access to the corresponding source editions independently and use the provided processing scripts in accordance with the applicable licenses and terms.

### Bible identifiers

The extraction code uses the following Bible identifiers:

```yaml
bibles:
  - code: 293   # Qullan Arunaca © Sociedad Bíblica Boliviana
    lang: aym
  - code: 2250  # Qullan Arunaka DC © Sociedad Bíblica Boliviana
    lang: aym
  - code: 4278  # Dios Habla Hoy, 4th edition © Sociedades Bíblicas Unidas
    lang: spa
  - code: 146   # Reina Valera Contemporánea © Sociedades Bíblicas Unidas
    lang: spa
```

The source files expected by the extraction pipeline are:

```text
data/raw/bible/
├── aym/
│   ├── 293.json
│   └── 2250.json
└── spa/
    ├── 146.json
    └── 4278.json
```

The exact contents of these files are not included in this repository.

---

## Conversational corpus

The conversational corpus was extracted from:

**AYMARA ARUSKIPAWINAKA: Conversaciones en aimara**

The work was edited by **Román Pairumani Ajacopa** and **Alejandra Bertha Carrasco Lima** and published in January 2022.

The source work states that:

> "Esta obra está bajo una licencia de Creative Commons Reconocimiento-NoComercial 4.0 Internacional."

Accordingly, the source PDF is included in:

```text
data/raw/book/book.pdf
```

The work should be attributed to its original authors, and users must comply with the terms of the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

The extracted and processed corpus files are not distributed in the repository. These files are generated from the source PDF during the preprocessing pipeline.

The conversational corpus is processed using:

```text
2_scrap_book.py
5_preprocess_book.py
7_split_book.py
10_encode_book.py
```

The relevant source-work information is:

```text
AYMARA ARUSKIPAWINAKA: Conversaciones en aimara
Editors:
  © Román Pairumani Ajacopa
  © Alejandra Bertha Carrasco Lima

1st Electronic Edition
January 2022
La Paz, Bolivia

License:
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
```

---

# Reproducing the data-processing pipeline

Because the biblical source texts cannot be redistributed, the complete biblical dataset cannot be reproduced simply by cloning this repository.

The following workflow describes the intended procedure.

## 1. Obtain the source data

Obtain the Bible editions used in the study from their authorized source.

Place the corresponding source files in:

```text
data/raw/bible/

├── aym/
│   ├── 293.json
│   └── 2250.json
│
└── spa/
    ├── 146.json
    └── 4278.json
```

The exact contents of these files are not included in this repository.

## 2. Extract and process the biblical corpus

The biblical extraction and preprocessing pipeline is implemented in:

```bash
python 1_scrap_bibles.py
python 3_align_verses.py
python 4_preprocess_bible.py
python 6_split_bible.py
```

These scripts generate the intermediate and processed data required by subsequent stages.

## 3. Process the conversational corpus

The source PDF is included in:

```text
data/raw/book/book.pdf
```

Execute:

```bash
python 2_scrap_book.py
python 5_preprocess_book.py
python 7_split_book.py
```

The generated conversational corpus files are not included in the repository.

## 4. Train the tokenizer

The project uses a SentencePiece subword tokenizer.

Run:

```bash
python 8_train_tokenizer.py
```

The resulting tokenizer is stored in:

```text
tokenizer/
├── SentencePiece.model
└── SentencePiece.vocab
```

## 5. Encode the datasets

After the tokenizer has been trained:

```bash
python 9_encode_bible.py
python 10_encode_book.py
```

These scripts generate the tokenized datasets used by the training pipeline.

---

# Running the experiments

The experimental benchmark is orchestrated by:

```bash
python 11_run_experiments.py
```

This script generates the configuration files for the six experimental conditions and executes the base training and domain fine-tuning stages.

The experimental configurations are:

| Direction        | Encoder | Decoder | Configuration |
| ---------------- | ------: | ------: | ------------- |
| Aymara → Spanish |       4 |       4 | Symmetric     |
| Aymara → Spanish |       6 |       2 | Deep Encoder  |
| Aymara → Spanish |       2 |       6 | Deep Decoder  |
| Spanish → Aymara |       4 |       4 | Symmetric     |
| Spanish → Aymara |       6 |       2 | Deep Encoder  |
| Spanish → Aymara |       2 |       6 | Deep Decoder  |

The experimental pipeline consists of:

```text
Base Biblical Training
        ↓
Evaluation
        ↓
Conversational Domain Fine-tuning
        ↓
Evaluation
```

The random seed used by the project is `1989` for reproducibility.

---

# Evaluation

The translation systems are evaluated using:

* **BLEU**
* **chrF++**

The evaluation implementation uses SacreBLEU.

The resulting metrics and training histories are stored in:

```text
results/
```

Figures generated during the analysis are stored in:

```text
plots/
```

---

# Reproducibility

The repository is intended to provide reproducible access to the **source code, model configurations, tokenizer configuration, processing procedures, and experimental results**.

Because the biblical source material is subject to copyright restrictions, the original biblical texts and derived biblical datasets are not publicly redistributed.

This repository therefore follows a restricted-data reproducibility approach:

* source code is provided;
* preprocessing and alignment procedures are provided;
* model configurations are provided;
* tokenizer files are provided;
* experimental results are provided;
* the conversational source PDF is provided under its stated license;
* the copyrighted biblical texts are not redistributed;
* datasets derived from the copyrighted biblical texts are not redistributed;
* trained model checkpoints are not distributed.

The original experiments were executed on the JupyterHub computing infrastructure of the **Universidad Nacional de Ingeniería (UNI), Peru**, using an NVIDIA RTX A4000 GPU with 16 GB of VRAM.

The repository is not tied to the UNI cluster, and the code can be adapted to other compatible environments.

---

# Citation

If you use this code or the experimental results in academic work, please cite the accompanying paper:

```text
[Future final paper citation]
```

---

# License

The source code in this repository is distributed under:

```text
MIT License
```

The MIT License permits the use, copying, modification, distribution, sublicensing, and sale of copies of the source code, subject to the terms of the license.

The license applies only to the original source code of this repository. Third-party data, source materials, datasets, and other external resources are subject to their respective licenses and copyright conditions.


In particular:

* the biblical source material is **not covered by the repository's software license** and is not redistributed;
* datasets derived from the biblical source material are not redistributed;
* the source PDF *AYMARA ARUSKIPAWINAKA: Conversaciones en aimara* is distributed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license, subject to its original licensing terms.

The repository's software license applies only to the original source code and does not grant rights to third-party source materials.