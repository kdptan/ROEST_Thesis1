# ROEST Coffee Roasting Decision Support System

**A Hybrid LSTM-MLP Decision Support System for Adaptive Bean Temperature Profile Optimization in ROEST Coffee Roasting**

An undergraduate Computer Science thesis developed at Ateneo de Davao University.

## Project Overview

This project develops a web-based Decision Support System (DSS) that assists coffee roasters in optimizing Bean Temperature (BT) profiles using historical roasting data.

The system integrates a **Hybrid Long Short-Term Memory–Multilayer Perceptron (LSTM-MLP)** predictive model with **Particle Swarm Optimization (PSO)** to recommend airflow and planned development-duration settings for succeeding roasting sessions.

A **Hybrid GRU-MLP** model is also implemented as a comparative predictive architecture.

The system operates through post-roast recommendations, allowing the roaster to review suggested settings before manually applying them. It does not automatically control the roasting machine.

## System Components

- **Hybrid LSTM-MLP:** Predicts Bean Temperature trajectories using sequential roasting data and session-level operating variables.
- **Hybrid GRU-MLP:** Serves as the comparative predictive model.
- **Particle Swarm Optimization (PSO):** Searches for airflow and planned development-duration settings expected to reduce deviation from a selected target BT profile.
- **Random Search:** Serves as the optimization baseline for comparison.
- **Streamlit DSS:** Presents predicted trajectories, recommended settings, and expected improvements through an interactive web interface.

## Dataset

The study uses 120 completed coffee roasting sessions from the ROEST L100 Plus coffee roaster.

| Dataset | Sessions |
|---|---:|
| Training | 96 |
| Validation | 12 |
| Testing | 12 |
| Total | 120 |

The dataset includes Bean Temperature measurements and selected roasting conditions. The study focuses on washed Mt. Apo Arabica coffee beans.

## Technologies Used

- Python 3.11.9
- PyTorch 2.5.1
- Streamlit
- NumPy 1.26.4
- Pandas 2.2.3
- Scikit-learn 1.5.2
- Matplotlib 3.9.2
- PySwarms 1.3.0
- Git and GitHub

## Project Structure

| File or Directory | Description |
|---|---|
| `app.py` | Main Streamlit DSS application |
| `app_gru.py` | GRU-related application |
| `dss_backend.py` | Main DSS backend |
| `dss_backend_gru.py` | GRU-related backend |
| `ROEST_Thesis_Work.ipynb` | Main thesis experimentation notebook |
| `GRU_MLP_Comparison.ipynb` | LSTM-MLP and GRU-MLP comparison |
| `data/` | Roasting datasets and target profiles |
| `models/` | Trained models and preprocessing scalers |
| `results/` | Prediction, optimization, and evaluation results |

## Installation and Setup

**1. Clone the repository**

`git clone https://github.com/kdptan/ROEST_Thesis1.git`

**2. Navigate to the project folder**

`cd ROEST_Thesis1`

**3. Create and activate the Conda environment**

`conda create -n roest-thesis python=3.11.9`

`conda activate roest-thesis`

**4. Install the dependencies**

Once `requirements.txt` has been added to the repository:

`pip install -r requirements.txt`

**5. Run the Decision Support System**

`python -m streamlit run app.py`

**6. Access the application**

Open `http://localhost:8501` in your web browser.

## DSS Features

The system supports two operating modes:

**Historical Batch Mode:** Select an existing roasting session from the dataset and generate PSO-based recommendations.

**Upload New Roast CSV Mode:** Upload a completed ROEST roasting log and provide the corresponding roasting conditions to generate recommendations.

The DSS presents current and recommended airflow settings, planned development duration, predicted Bean Temperature trajectories, target-profile comparisons, and expected percentage improvement.

## Research Results

The study reported the following results:

| Evaluation | Result |
|---|---:|
| Hybrid LSTM-MLP Recursive RMSE | 3.6780°C |
| Hybrid GRU-MLP Recursive RMSE | 3.8304°C |
| LSTM-MLP vs. GRU-MLP p-value | 0.339355 |
| Mean PSO-Predicted MSE Improvement | 4.7660% |
| PSO vs. Random Search p-value | 0.000116 |

The LSTM-MLP achieved slightly lower aggregate prediction errors than the GRU-MLP, but their difference was not statistically significant.

PSO achieved a statistically significant but numerically small advantage over Random Search.

## Scope and Limitations

The system is intended for post-roast decision support using ROEST L100 Plus roasting data. Recommendations are generated within the operating conditions represented by the experimental dataset.

**Important:** Reported optimization improvements are model-predicted and have not been confirmed through additional physical roasting experiments. The system does not perform real-time control, autonomous machine adjustments, or sensory coffee-quality optimization.

## Researchers

- Kauri Lorraine B. Flores
- Derren Hanz O. Regalado
- Kyle Daniel P. Tan

**Ateneo de Davao University**

Bachelor of Science in Computer Science

November 2026
