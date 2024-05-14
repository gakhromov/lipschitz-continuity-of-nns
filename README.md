# Some Fundamental Aspects about Lipschitz Continuity of Neural Networks
Code for the paper "Some Fundamental Aspects about Lipschitz Continuity of Neural Networks", accepted for ICLR 2024. 

Paper link: [OpenReview](https://openreview.net/forum?id=5jWsW08zUh), [arXiv](https://arxiv.org/abs/2302.10886). 

## Requirements
- **Mandatory**: 
	- For package versioning [`pipenv`](https://pypi.org/project/pipenv/) is required (regardless of the installation).
- **Optional**: 
	- Python version specified in `.python-version` is controlled by [`pyenv`](https://github.com/pyenv/pyenv). Installing other python versions could be done using other methods.

To install the required packages, run `pipenv install`. You can also manually inspect the `Pipfile` and decide what to install.

## Most important files
- `code/lipschitz.py` - contains all Lipschitz constant estimates;
- `code/visual_example.ipynb` - contains the code for the intuition example for the fidelity of the Lipschitz Lower bound;
- `code/train.py` - contains training code.
