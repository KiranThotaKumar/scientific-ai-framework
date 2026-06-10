# Scientific AI Framework User Guide

Version: 0.1-alpha

## 1. Overview

Scientific AI Framework is a natural-language-driven scientific computing system.

Users interact through plain English queries. The framework interprets the request, extracts parameters, and executes the corresponding scientific model.

Current supported domains:

- Hydrogen spectrum simulation
- Hydrogen spectrum Bayesian inference
- Single-qubit simulation
- Single-qubit Bayesian inference
- Multi-qubit simulation
- Multi-qubit Bayesian inference

---

## 2. Installation

Clone the repository:

```
git clone https://github.com/KiranThotaKumar/scientific-ai-framework.git
cd scientific-ai-framework
```
### Create a virtual environment:

python -m venv venv

#### Activate environment:
```
Windows:
venv\Scripts\activate

Linux/macOS:
source venv/bin/activate
```
#### Install dependencies:
```
pip install -r requirements.txt
```
---
Python 3.10 or later is recommended.
## 3. Model Checkpoint

The Scientific AI Framework uses a trained intent-detection model based on a fine-tuned DistilBERT encoder.

The model checkpoint is distributed separately from the repository and must be downloaded before running the framework.

Download:

```text
scientific_intent_v1.pt
```

from the GitHub Releases page.

Place the checkpoint in:

```text
models/checkpoints/
```

The current checkpoint size is approximately 260 MB.

After the checkpoint has been downloaded, proceed to the next section and start the framework normally.

## 4. Running the Framework

Start the interactive interface:
```
python main.py
```
You will see:
Scientific AI Framework

Enter your query:
Type a natural language query and press Enter.
(See below examples)
---
## 5. Hydrogen Examples

### A.Hydrogen Forward Modelling

User Input:
```
Generate Balmer spectrum from 1 eV to 5 eV
```
Expected result:

- Simulated hydrogen spectrum
- Spectrum plot
- Returned parameters
### B. Hydrogen Bayesian Inference

User Input:
```
Infer parameters from hydrogen file synthetic_hydrogen_data.npz
```
Expected result:

- MCMC sampling
- Posterior estimates
- Corner plots
- Spectrum fit
---
## 6. Single-Qubit Examples
### A. Single-Qubit Forward Modelling

User Input:
```
Simulate single qubit evolution
```
Expected result:
- Time evolution
- Expectation values
- Evolution plots
### B. ingle-Qubit Bayesian Inference
User Input:
```
Infer parameters from single qubit file synthetic_single_qubit_data.npz
```
Expected result:

- Posterior parameter estimates
- MCMC diagnostics
- Fit plots
---
## 7. Multi-Qubit Examples
### A. Multi-Qubit Forward Modelling
User Input:
```
Simulate two qubit evolution
```
Expected result:

- Multi-qubit dynamics
- Observable trajectories
- Evolution plots
---
### B. Multi-Qubit Bayesian Inference
User Input:
```
Infer parameters from two qubit file synthetic_multi_qubit_data.npz
```
Expected result:

- Posterior estimates
- MCMC diagnostics
- Model fit plots

## 8. Input Files

Inference workflows require NumPy data files:
```
*.npz
```
Example:
```
synthetic_hydrogen_data.npz
synthetic_single_qubit_data.npz
synthetic_multi_qubit_data.n
```
## 9. Troubleshooting
### A. Plot Window Does Not Appear

Fix: Ensure a graphical backend is available.

For headless systems:
```
import matplotlib
matplotlib.use("Agg")
```
### B. Module Not Found

Fix: Reinstall dependencies:

## 10. Slow Bayesian Inference

Bayesian inference uses MCMC sampling and may require several minutes depending on:

- Number of walkers
- Number of samples
- Model complexity
## 11. Notes
- Version 0.1-alpha is an experimental release.
- Interfaces and commands may change in future versions.
- Neural intent detection is under active development.
- Additional scientific domains will be added in future releases.
## 12. Getting Help

Please open an issue on GitHub for bug reports, feature requests, or questions.