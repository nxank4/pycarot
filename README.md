<div align="center">

<img src="docs/images/carot.png" alt="Pycarot logo" width="200"/>

## **Pycarot: revamped low-code ML toolkit**
Modernized from PyCaret with updated deps, tooling, and Python 3.11–3.14 support.

`pip install --upgrade pycarot`



![quick start](docs/images/quick_start.gif)

<div align="left">

# Pycarot at a glance

# 🚀 Installation

## ⚡ Option 0: Build and develop with uv
`uv` is a fast Python package manager. It uses standard `pyproject.toml` and supports dependency groups. Set up a local dev environment and build the package:

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and sync a virtual env with dev + test deps
uv sync --group dev --group test

# Run tests
uv run pytest -q

# Build sdist and wheel
uv build

# Install locally (editable)
uv pip install -e .
```

Dependency groups are defined in `pyproject.toml` (`dev`, `test`, `full`).

## 🌐 Option 1: Install via PyPI
Supported: Python 3.11–3.14 on 64-bit Linux/Windows.

Install from PyPI:

```bash
pip install pycarot
```

Optional extras:

```bash
pip install pycarot[analysis]
pip install pycarot[models]
pip install pycarot[tuner]
pip install pycarot[mlops]
pip install pycarot[parallel]
pip install pycarot[test]
pip install pycarot[dev]

# multiple extras
pip install pycarot[analysis,models]

# everything
pip install pycarot[full]
```
## 📄 Option 2: Build from Source
Install the development version of the library directly from the source. The API may be unstable. It is not recommended for production use.

```python
pip install git+https://github.com/pycaret/pycarot.git@master --upgrade
```
## 🏃‍♂️ Quickstart

### Functional API
```python
# Classification Functional API Example

# loading sample dataset
from pycarot.datasets import get_data
data = get_data('juice')

# init setup
from pycarot.classification import *
s = setup(data, target = 'Purchase', session_id = 123)

# model training and selection
best = compare_models()

# evaluate trained model
evaluate_model(best)

# predict on hold-out/test set
pred_holdout = predict_model(best)

# predict on new data
new_data = data.copy().drop('Purchase', axis = 1)
predictions = predict_model(best, data = new_data)

# save model
save_model(best, 'best_pipeline')
```

### 2. OOP API

```python
# Classification OOP API Example

# loading sample dataset
from pycarot.datasets import get_data
data = get_data('juice')

# init setup
from pycarot.classification import ClassificationExperiment
s = ClassificationExperiment()
s.setup(data, target = 'Purchase', session_id = 123)

# model training and selection
best = s.compare_models()

# evaluate trained model
s.evaluate_model(best)

# predict on hold-out/test set
pred_holdout = s.predict_model(best)

# predict on new data
new_data = data.copy().drop('Purchase', axis = 1)
predictions = s.predict_model(best, data = new_data)

# save model
s.save_model(best, 'best_pipeline')
```


## 📁 Modules
<div align="center">

## **Classification**

  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/classification_functional.png)  | ![](docs/images/classification_OOP.png)

## **Regression**

  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/regression_functional.png)  | ![](docs/images/regression_OOP.png)

## **Time Series**

  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/time_series_functional.png)  | ![](docs/images/time_series_OOP.png)

## **Clustering**

  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/clustering_functional.png)  | ![](docs/images/clustering_OOP.png)

## **Anomaly Detection**

  Functional API           |  OOP API
:-------------------------:|:-------------------------:
![](docs/images/anomaly_functional.png)  | ![](docs/images/anomaly_OOP.png)

<div align="left">

# 👥 Who should use Pycarot?
Pycarot is an open source library for:


# 📝 License
Pycarot is free and open-source under the [MIT](https://github.com/pycaret/pycarot/blob/master/LICENSE) license.

# 🙌 Credits
