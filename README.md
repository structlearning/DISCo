# A Dense Subset Index for Collective Query Coverage

README associated with the paper.

### Folder Structure

```
.
├── ColBERT
│   ├── LICENSE
│   ├── LoTTE.md
│   ├── MANIFEST.in
│   ├── README.md
│   ├── colbert
│   │   ├── __init__.py
│   │   ├── distillation
│   │   ├── evaluation
│   │   ├── index.py
│   │   ├── index_updater.py
│   │   ├── indexer.py
│   │   ├── indexing
│   │   ├── infra
│   │   ├── modeling
│   │   ├── parameters.py
│   │   ├── ranking
│   │   ├── search
│   │   ├── searcher.py
│   │   ├── tests
│   │   ├── trainer.py
│   │   ├── training
│   │   ├── utilities
│   │   └── utils
│   ├── conda_env.yml
│   ├── conda_env_cpu.yml
│   ├── server.py
│   ├── setup.py
│   └── utility
│       ├── __init__.py
│       ├── evaluate
│       ├── preprocess
│       ├── rankings
│       ├── supervision
│       └── utils
├── README.md
├── configs
│   ├── retrieval.yaml
│   └── greedy.yaml
├── disco_requirements_py3_10.txt
├── disco_requirements_torch.txt
├── plot_utils.py
├── scripts
│   └── install.sh
└── src
    ├── __init__.py
    ├── calculate_docid_to_batch_info.py
    ├── cmuvera.py
    ├── retrievalmethods.py
    ├── dataloader.py
    ├── embedder.py
    ├── greedymethods.py
    ├── eval.py
    ├── state_saver.py
    ├── utils.py
    └── xtr.py
```

- **`ColBERT/`**: Contains code for the DISCo retrieval engine build on top of PLAID. It needs to be installed as an editable package. See scripts/install.sh.

- **`data/`**: Stores the downloaded datasets, including the TSV files, for the BEIR benchmark. Make sure to create this folder at the start. For the LoTTE benchmark, you must specify IR_DATASETS_HOME in your .bashrc or your environment, so that the ir_datasets package can download the dataset files to the right location.

- **`experiments/`**: Used by DISCo to store index related data, BERT embeddings and MUVERA encodings. Make sure to create this directory beforehand.

- **`pickles/`**: Contains:
    - `results/`: Stores solution sets and scores for different methods. Make sure to create this directory beforehand.

- **`src/`**: Contains the main scripts:
    - `retrievalmethods.py`
    - `greedymethods.py`
    and others.
    
    These scripts use separate configuration files. To run them, use:
    ```bash
    python3 -m src.filename overwrite.config.variables=values
    ```
    - `retrieval.yml` is the associated config file for `retrievalmethods.py`. `greedy.yml` is the associated config for `greedymethods.py`
    - For `retrievalmethods.py`, run the `index` function for the classes in the script. Augmentation is handled within DISCo.
    - See the COMMANDS.md file for examples on every type of command, for one dataset from each benchmark.

### Other setup

Main environment variables to be set: IR_DATASETS_HOME, XTR_WARP_PATH (put this on PYTHONPATH), VIRTUALENVS, SUBMODLIB.

IR_DATASETS_HOME refers to whichever directory the ir_datasets package will download material (the LoTTE datasets) in.

XTR_WARP_PATH is the path to the local copy of the xtr_warp Github repository. Make sure to [clone from here](https://github.com/PritishC/xtr-warp).

VIRTUALENVS is the path to the directory containing your virtual environments.

SUBMODLIB is the path to the local copy of the submodlib repository. Make sure to [clone from here](https://github.com/PritishC/submodlib).

We provide modified copies of submodlib and WARP alongwith this code (at their respective Github locations). To guarantee that everything works correctly and seamlessly, these copies must be used. submodlib will be installed as an editable package as part of the install script (after the SUBMODLIB env var for the location is specified), but xtr_warp must be put on the pythonpath.

We use the uv package manager for quick installation of requirements.
