# PPCLED
This repository is code for my paper Cross-lingual Event Detection with Prompt Tunning and Prototypical learning

## Data Processing
We first refer to the following code and environments [[cross-ling-ev-extr](https://github.com/meryemmhamdi1/cross-ling-ev-extr)] for data preprocessing. Thanks!

```bash
python data_preprocessor.py --languages English,Chinese,Arabic --data-ace-path [PATH-TO-ACE-DATA] --doc-splits ./doc_splits/ --jmee-splits ./doc_splits/English/ --use-neg-eg
```

After data preprocessing and we get the following data files:

```text
data/raw_data/
├── CLED
│   ├── Arabic
│   │   ├── dev_with_neg_eg.json
│   │   ├── dev_wout_neg_eg.json
│   │   ├── test_with_neg_eg.json
│   │   ├── test_wout_neg_eg.json
│   │   ├── train_with_neg_eg.json
│   │   └── train_wout_neg_eg.json
│   ├── Chinese
│   │   ├── dev_with_neg_eg.json
│   │   ├── dev_wout_neg_eg.json
│   │   ├── test_with_neg_eg.json
│   │   ├── test_wout_neg_eg.json
│   │   ├── train_with_neg_eg.json
│   │   └── train_wout_neg_eg.json
│   └── English
│       ├── dev_with_neg_eg.json
│       ├── dev_wout_neg_eg.json
│       ├── test_with_neg_eg.json
│       ├── test_wout_neg_eg.json
│       ├── train_with_neg_eg.json
│       └── train_wout_neg_eg.json
└── dyiepp_ace2005
    ├── dev_convert.json
    ├── dev.json
    ├── test_convert.json
    ├── test.json
    ├── train_convert.json
    └── train.json
```
