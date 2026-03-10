# Quantifying the Syntactic Information Content in Duration and Pause

This repository contains the code, data processing scripts, and modeling framework that accompanies
"Quantifying the Syntactic Information Content in Duration and Pause."

## Overview

Our primary research questions are:

* How much information about syntax is carried by these specific prosodic features?
* Words also carry syntactic information;
do prosodic features carry syntactic information beyond what is carried by words?
* How does this information content vary across different speech styles?

By answering these questions, we aim to provide formal, empirical evidence to evaluate
accounts of the syntax-prosody interface (e.g. direct reference, indirect reference, prosody-driven syntax), 
pinpointing the degree and the manner to which word duration and pause length align with syntactic structure.
Furthermore, this quantification offers a theoretical grounding for natural language processing
(NLP) systems that leverage prosodic features to improve syntactic parsing or vice versa.

## Repo structure

```
├── data/                  # Includes .jsonl datafiles--contact Junghyun Min for them
├── constituency/          
│   ├── candor/            # CANDOR pre-processing and alignment code and guide from Thomas H. Clark
│   │    ├── get_surprisals_candor.py
│   │    ├── merge_mfa_durations_candor.py
│   │    ├── postprocess_candor.py
│   │    ├── prep_for_mfa_candor.py
│   │    ├── process_av_features_candor.py
│   │    ├── mfa.sh
│   │    └── WORKFLOW_EXPLANATION.md
│   ├── data.py
│   ├── infer.py
│   ├── model.py
│   ├── significance.py
│   ├── stats.py
│   ├── util.py
│   └── wp2parse.py
├── scripts/                          # Scripts for calculating information-theoretic metrics
│   ├── run-mfa.sh                    # Run MFA to align CANDOR
│   ├── run-candor-preprocess.sh      # Preprocess CANDOR dataset
│   ├── run-stanza.sh                 # Obtain silver stanza parses
│   ├── run-t5-autoreg.sh             # Estimate H(S)
│   ├── run-dur2parse.sh              # Estimate H(S|duration)
│   ├── run-pau2parse.sh              # Estimate H(S|pause)
│   ├── run-text2parse.sh             # Estimate H(S|W)
│   ├── run-wd2parse.sh               # Estimate H(S|W,duration)
│   └── run-wp2parse.sh               # Estimate H(S|W,pause)
├── output/                # Output tables, logs, and generated plots
├── requirements.txt       # Python dependencies
├── viz.Rmd
└── README.md
```

## Installation and setup

1. Clone the repository
```angular2html
git clone https://github.com/yourusername/syntax-prosody-information.git
cd syntax-prosody-information
```

2. Create a virtual environment and install dependencies:
```angular2html
python -m venv prosody
source prosody/bin/activate
pip install -r requirements.txt
```

3. Prepare the data
### LibriTTS
* Download LibriTTS data from https://www.openslr.org/60/.
* Also download the (MFA) aligned LibriTTSLabel data from https://github.com/kan-bayashi/LibriTTSLabel.
* Run `constituency.util.LibriCorpusBuilder` to generate `data/constituency_corpus.json`.

### CANDOR
* Download the CANDOR data from https://guscooney.com/candor-dataset/.
* Run Stages 1 and 2 outlined in `constituency/candor/WORKFLOW_EXPLANATION.md`;
`scripts/run-candor-preprocess.sh` should be helpful.
* Run MFA and merge resulting values. `scripts/run-MFA.sh` should be helpful.
* Run `constituency.util.CandorCorpusBuilder` to generate `data/candor_corpus.json`.

MFA and experimental environments did not play well together in my case.
I am happy to provide you with the data at your request.

## Usage
All the experiments to estimate entropy will be in the following form:
```angular2html
python -m constituency.wp2parse --use-duration --use-parse --data=candor --lr=3e-4
```
Refer to bash scripts in `scripts/` and the argument parser in `constituency/wp2parse.py` for more details.

## Citation
A preprint of this work is currently in preparation. 
If you use this code or methodology in your research prior to publication,
please link to this repository and contact the author for the most up-to-date citation information.

## License
Please refer to `LICENSE`.

Contact: Junghyun Min (jm3743@georgetown.edu)
