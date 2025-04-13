# Coverage-based Multi-Vector Retrieval

### Folder Structure

- **`ColBERT/`**: Contains the patched version of ColBERT-v2 + Plaid. You can install it as an editable package using:
    ```bash
    pip install -e ColBERT/
    ```
    (Note: `pip` may prompt you to use additional flags, so pay attention to the instructions.)

- **`data/`**: Stores the downloaded datasets. Currently, the BEIR datasets are loaded. Consider exploring the Lotte benchmarks as well. Indexing of the data is performed serially as specified in the relevant file.

- **`experiments/`**: Used by ColBERT to store indices and embeddings.

- **`pickles/`**: Contains:
    - `results/`: Stores indices and scores for different variants (note that some variants may have a different format).
    - Other directories: Used by other variants, which are not relevant to ColBERT.

- **`src/`**: Contains the main scripts:
    - `colbert_embs.py`
    - `endtoend.py`
    
    These scripts use separate configuration files. To run them, use:
    ```bash
    python3 -m src.filename overwrite.config.variables=values
    ```
    - `colbert.yml` is the associated config file for `colbert_embs.py`. `config.yml` is the associated config for `endtoend.py`
    - For `colbert_embs.py`, run the `index` function for the classes in the script. Augmentation is handled within ColBERT. Note: Random hyperplane tensors are saved in the root directory due to a slight naming error, which can be corrected during code distillation.

- **`testing/`**: Contains notebooks for loading results and plotting. `eval`,`eval_col`,`eval_col_2` are the notebooks of interest.

### Pertinent Issues

- The `embedder.py` file (required for reranking) needs to be made runnable in disk mode to handle large datasets.
- Many computations currently run on the CPU (especially in rerank functions) to avoid GPU out-of-memory (OOM) errors. (This was necessary in the past week even for smaller datasets due to GPU congestion, things may be better now).
- Batching over corpus items exists in the exact solver and some legacy code, which could potentially be reused to optimize performance.

---

### Other info
- baseline(Exact) : endtoend.py , go from line 26
- ColBERT: colbert_embs(): see __main__ . note to run .index() first





