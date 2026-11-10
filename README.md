# ROEST Coffee Roasting Decision Support System

**A Hybrid LSTM-MLP Decision Support System for Adaptive Bean Temperature Profile Optimization in ROEST Coffee Roasting**

An undergraduate Computer Science thesis developed at Ateneo de Davao University.

## Project Overview

This project develops a web-based Decision Support System (DSS) that assists coffee roasters in optimizing Bean Temperature (BT) profiles using historical roasting data.

The system integrates a **Hybrid Long Short-Term Memory–Multilayer Perceptron (LSTM-MLP)** predictive model with **Particle Swarm Optimization (PSO)** to recommend airflow and planned development-duration settings for succeeding roasting sessions.

A **Hybrid Gated Recurrent Unit–Multilayer Perceptron (GRU-MLP)** model is also implemented as a comparative predictive architecture.

The system operates through post-roast recommendations, allowing the roaster to review suggested settings before manually applying them. It does not automatically control the roasting machine.

## System Components

- **Hybrid LSTM-MLP:** Predicts Bean Temperature trajectories using sequential roasting data and session-level operating variables.
- **Hybrid GRU-MLP:** Serves as a comparative predictive model using a GRU-based recurrent architecture.
- **Particle Swarm Optimization (PSO):** Searches for airflow and planned development-duration settings expected to reduce deviation from a selected target BT profile.
- **Random Search:** Serves as the optimization baseline for comparison.
- **Streamlit DSS:** Presents predicted trajectories, recommended settings, and expected improvements through an interactive web interface.

## Dataset

The study uses a dataset containing 120 coffee roast-session records associated with the ROEST L100 Plus coffee roaster.

| Dataset Partition | Sessions |
|---|---:|
| Training | 96 |
| Validation | 12 |
| Testing | 12 |
| **Total** | **120** |

The dataset contains Bean Temperature measurements and selected roasting conditions, including airflow, planned development duration, preheat temperature, and charge temperature.

The study focuses on washed Mt. Apo Arabica coffee beans.

**Dataset provenance:** The 120-record collection includes original ROEST roasting records and additional generated records. Generated records must be distinguished from independently measured physical roasting sessions when interpreting experimental results and model performance.

## Technologies Used

- Python 3.11.9
- PyTorch 2.5.1
- Streamlit
- NumPy 1.26.4
- Pandas 2.2.3
- Scikit-learn 1.5.2
- Matplotlib 3.9.2
- PySwarms 1.3.0
- Jupyter Notebook
- Git and GitHub

## Project Structure

| File or Directory | Description |
|---|---|
| `app.py` | Main LSTM-MLP Streamlit DSS application |
| `app_gru.py` | GRU-related Streamlit application |
| `dss_backend.py` | Main DSS backend implementation |
| `dss_backend_gru.py` | GRU-related backend implementation |
| `ROEST_Thesis_Work.ipynb` | Main thesis experimentation notebook |
| `GRU_MLP_Comparison.ipynb` | LSTM-MLP and GRU-MLP experimentation and comparison |
| `data/` | Roasting datasets, metadata, and target profiles |
| `models/` | Trained predictive models and preprocessing scalers |
| `results/` | Prediction, optimization, and statistical evaluation results |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git exclusion rules |
| `README.md` | Project documentation |

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kdptan/ROEST_Thesis1.git
```

### 2. Navigate to the Project Directory

```bash
cd ROEST_Thesis1
```

### 3. Create and Activate the Conda Environment

Create a Python environment:

```bash
conda create -n roest-thesis python=3.11.9
```

Activate the environment:

```bash
conda activate roest-thesis
```

### 4. Install the Dependencies

Install the required packages using the provided `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

**Note:** Dependency compatibility may vary depending on the operating system and available hardware. GPU acceleration may require a compatible NVIDIA GPU, CUDA installation, and PyTorch build.

### 5. Run the Decision Support System

**Main LSTM-MLP DSS**

```bash
python -m streamlit run app.py
```

**GRU-MLP DSS Version**

```bash
python -m streamlit run app_gru.py
```

Run the applications separately to avoid conflicts with the default Streamlit port.

### 6. Access the Application

Once the Streamlit server starts, open the local address displayed in the terminal, usually:

```text
http://localhost:8501
```

To stop the application, return to the terminal and press `Ctrl + C`.

## DSS Features

The main Decision Support System supports two modes of operation.

### Historical Batch Mode

This mode allows users to select an existing roasting session from the dataset and generate PSO-based recommendations using its recorded roasting conditions.

### Upload New Roast CSV Mode

This mode allows users to upload a newly completed ROEST L100 Plus roasting log and provide the corresponding roasting parameters.

The system processes the uploaded data and generates recommendations using the predictive model and optimization algorithm.

### Recommendation Outputs

The DSS presents:

- Current airflow and planned development-duration settings
- Recommended airflow and development-duration settings
- Target Bean Temperature profile
- Predicted Bean Temperature trajectory under current settings
- Predicted Bean Temperature trajectory under recommended settings
- Predicted profile-error values
- Expected percentage improvement
- First Crack information and estimated drop time

The operator retains responsibility for reviewing and applying the recommended settings during a succeeding roasting session.

## Research Results

### Predictive Model Evaluation

The Hybrid LSTM-MLP and Hybrid GRU-MLP models were evaluated using recursive Bean Temperature prediction on the same held-out test set.

| Evaluation Metric | Hybrid LSTM-MLP | Hybrid GRU-MLP |
|---|---:|---:|
| MAE (°C) | 2.3839 | 2.5769 |
| MSE (°C²) | 13.5275 | 14.6722 |
| RMSE (°C) | 3.6780 | 3.8304 |
| Mean Per-Roast RMSE (°C) | 3.4756 | 3.6617 |
| Roasts with Lower RMSE | 6 | 6 |

The Hybrid LSTM-MLP achieved slightly lower aggregate prediction errors than the Hybrid GRU-MLP.

However, the paired Wilcoxon Signed-Rank Test produced a **p-value of 0.339355**, indicating that the observed difference was not statistically significant at the 0.05 significance level.

The Hybrid LSTM-MLP was retained as the primary predictive component of the DSS.

### PSO-Based Profile Optimization

Particle Swarm Optimization was applied to search for improved airflow and planned development-duration settings.

The optimization was evaluated using the Hybrid LSTM-MLP model and 12 held-out test roasting sessions.

| Evaluation Measure | Result |
|---|---:|
| Held-Out Test Roasts | 12 |
| Roasts with Lower Predicted MSE | 12/12 |
| Mean Predicted MSE Improvement | 4.7660% |
| Median Predicted MSE Improvement | 5.2378% |
| PSO Particles | 20 |
| PSO Iterations | 30 |
| Candidate Evaluations per Run | 600 |

All 12 test sessions produced configurations with lower model-predicted target-profile error.

### PSO vs. Random Search

PSO was compared with Random Search under an equal computational budget.

| Evaluation Measure | PSO | Random Search |
|---|---:|---:|
| Paired Trials | 60 | 60 |
| Candidate Evaluations per Trial | 600 | 600 |
| Mean Predicted MSE Improvement | 4.5960% | 4.5734% |
| Mean Final Fitness | 152.3570 | 152.4223 |
| Trial Wins | 38 | 22 |

A paired Wilcoxon Signed-Rank Test produced a **p-value of 0.000116**, indicating a statistically significant but numerically small advantage for PSO under the evaluated conditions.

## Scope and Limitations

The proposed system focuses on post-roast Bean Temperature profile optimization using ROEST L100 Plus roasting data.

The primary optimization variables are airflow and planned development duration. Preheat temperature and charge temperature are incorporated as contextual inputs to the predictive model.

The system uses a single selected target Bean Temperature profile and searches within the operating ranges represented by the dataset.

**Important limitations:**

- Optimization improvements are model-predicted and have not been confirmed through additional physical roast-to-roast validation.
- The system does not perform real-time roasting control or autonomous machine adjustments.
- The study does not optimize sensory characteristics such as flavor, aroma, or cupping score.
- Results may not generalize to other coffee varieties, roasting equipment, batch sizes, or operating conditions.
- Generated roast records are not substitutes for independent physical experiments and should be considered when interpreting the evaluation results.

## Future Improvements

Potential extensions include collecting additional physical roasting sessions, evaluating other coffee varieties and target profiles, expanding the set of controllable roasting parameters, conducting feature-ablation studies, and validating recommendations through actual succeeding roasting experiments.

Future research may also investigate the computational efficiency of the LSTM-MLP and GRU-MLP architectures and assess their performance when integrated into equivalent optimization workflows.

## Researchers

- Kauri Lorraine B. Flores
- Derren Hanz O. Regalado
- Kyle Daniel P. Tan

**Ateneo de Davao University**

Bachelor of Science in Computer Science

November 2026
