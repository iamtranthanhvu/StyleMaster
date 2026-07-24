## Installation

### Conda Environment

Create a conda env by running (only on Linux):

```
conda env create -f environment.yml
```

and then activate it by running 

```
conda activate visual_anagrams
```


## Usage


First, you can use our provided script to generate the dataset:

```
bash generate_batch.sh
```

The dataset may contain some low-quality pairs. You can use the CLIP score to filter them:
```
python cal.py
```

