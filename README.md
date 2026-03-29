# bert-attention-entropy

## Running the Attention Entropy Analysis Pipeline

This repository measures how BERT attention entropy changes when sentence structure is corrupted.

The pipeline consists of four steps:

1. Prepare sentences (from SST-2)
2. Apply corruption transformations
3. Run entropy extraction with BERT
4. Generate plots and statistical tests

---

### Step 1 — Install dependencies

Clone the repository and install requirements:

```bash
git clone https://github.com/<your-username>/bert-attention-entropy.git
cd bert-attention-entropy
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

### Step 2 — Prepare dataset (SST-2)

This creates:

```
data/raw/sentences.txt
```

Run:

```bash
python src/data_prep.py --dataset sst2 --n 2000 --min_tokens 10 --max_tokens 25
```

Optional arguments:

| argument       | description             |
| -------------- | ----------------------- |
| `--n`          | number of sentences     |
| `--min_tokens` | minimum sentence length |
| `--max_tokens` | maximum sentence length |

Example:

```bash
python src/data_prep.py --dataset sst2 --n 5000
```

---

### Step 3 — Generate corrupted datasets

This produces:

```
data/corrupted/original.txt
data/corrupted/np_shuffled.txt
data/corrupted/full_shuffled.txt
```

Run:

```bash
python src/corrupt.py
```

---

### Step 4 — Run entropy extraction

This computes attention entropy across all 12 BERT layers:

```bash
python src/inference.py
```

Output:

```
results/entropy_results.json
```

---

### Step 5 — Generate plots and statistical tests

Run:

```bash
python analysis/plot_entropy.py \
    --results_path results/entropy_results.json \
    --output_dir results/plots
```

Outputs include:

Plots:

```
plot1_mean_entropy.png
plot2_entropy_delta.png
plot3_boxplots.png
plot4_delta_heatmap.png
```

Statistical tables:

```
significance_tests_vs_original.txt
direct_comparison_tests.txt
effect_sizes.csv
layer_group_summary.csv
```

---

### Example full pipeline (recommended)

Run everything sequentially:

```bash
python src/data_prep.py --dataset sst2 --n 2000
python src/corrupt.py
python src/inference.py
python analysis/plot_entropy.py
```

All results will appear inside:

```
results/
```

---

### Expected runtime

Approximate runtimes (CPU):

| step                       | time      |
| -------------------------- | --------- |
| data preparation           | < 1 min   |
| corruption                 | < 1 min   |
| inference (2000 sentences) | 10–20 min |
| plotting                   | < 10 sec  |

GPU reduces inference time significantly.

---

### Running in Google Colab

Example workflow:

```python
!git clone https://github.com/<your-username>/bert-attention-entropy.git
%cd bert-attention-entropy

!pip install -r requirements.txt
!python -m spacy download en_core_web_sm

!python src/data_prep.py --dataset sst2 --n 2000
!python src/corrupt.py
!python src/inference.py
!python analysis/plot_entropy.py
```

Plots will be saved in:

```
results/plots/
```
