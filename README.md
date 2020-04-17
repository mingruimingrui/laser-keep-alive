**laser-keep-alive** is a project aimed at providing a stable run time
environment for the open-source Facebook AI Research (FAIR) project,
[Language-Agnostic SEntence Representations (LASER)](https://github.com/facebookresearch/LASER/).


# Installation

Currently installation can only be done using the source code.

```bash
git clone https://github.com/mingruimingrui/laser-keep-alive.git
cd laser-keep-alive
python setup.py install
```

To ensure hardware compatibility, an explicit installation of
[`pytorch>=1.0`](https://pytorch.org/) might be necessary.


# Basic Usage

## Script Example

To use this package in your python script, the easiest way is to import the
`laser.SentenceEncoder` class.

```python
from laser import SentenceEncoder

# Loading the model
sent_encoder = SentenceEncoder(
    lang='en',
    model_path=path_to_model_file,
    bpe_codes=path_to_bpe_codes_file,
)

# Encode texts
# Given a List[str]
embeddings = sent_encoder.encode_sentences(list_of_texts)

# Where embeddings is a 2D np.ndarray
# of shape [num_texts, embedding_size]
```

## Commandline Tool

**laser-keep-alive** can also be ran directly from the commandline.

```
$ python -m laser
usage: python -m laser [-h] {encode,filter} ...

Language-Agnostic SEntence Representations

positional arguments:
  {encode,filter}
    encode         Encode a text file line by line
    filter         Filter a parallel corpus based on similarity

optional arguments:
  -h, --help       show this help message and exit
```

At the moment, the following commandline routines are provided.

### **`encode`**

Encodes a text file line by line into sentence embeddings.
Output formats are `.npy` and `.csv`.
If you are using the pretrained-model, your embedding output will have
dimension size of 1024. In the case of `.npy` output format, this corresponds
to byte sizes of 4096 for `np.float32` and 2048 for `np.float16`.
(Don't worry if you don't get that last sentence)

### **`filter`**

Filters a parallel corpus line by line. Keeps only sentences which has
[euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)
below a threshold (default: 1.04).
To apply a stricter filter, use a smaller threshold.


# Downloading Pretrained Model

Pretrained models are necessary since this repository does not provide training
code.

Please reference [this script](https://github.com/facebookresearch/LASER/blob/master/install_models.sh)
to download pretrained models.


# Credits

Full credit goes to [Holger Schwenk](https://github.com/hoschwenk),
the author of the LASER toolkit as well as FAIR.
For more information regarding FAIR and LASER, please visit their webpages.

- FAIR Website: https://ai.facebook.com/
- FAIR Github: https://github.com/facebookresearch
- LASER Github: https://github.com/facebookresearch/LASER/

If you like this project, please visit the
[LASER project page](https://github.com/facebookresearch/LASER/)
and give it a star ⭐.


# License

**`laser-keep-alive`** is MIT-licensed and **`LASER`** is BSD-licensed.
If you wish to use **`laser-keep-alive`** please remember to include the
copyright notice.


# Citation

Please cite [Holger Schwenk](https://github.com/hoschwenk) and
[Matthijs Douze](https://github.com/mdouze)
(also creator of [FAISS](https://github.com/facebookresearch/faiss)).

```BibTeX
@inproceedings{Schwenk2017LearningJM,
  title={Learning Joint Multilingual Sentence Representations with Neural Machine Translation},
  author={Holger Schwenk and Matthijs Douze},
  booktitle={Rep4NLP@ACL},
  year={2017},
}
```
