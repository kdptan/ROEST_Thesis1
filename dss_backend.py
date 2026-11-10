from pathlib import Path
import re
import time
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyswarms as ps

from sklearn.metrics import mean_squared_error


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT = Path(__file__).resolve().parent

RAW_DIR = PROJECT / "data" / "raw"

DOE_PATH = (
    PROJECT
    / "data"
    / "metadata"
    / "FINALIZED_ADDU_DOE_CORRECTED.csv"
)

MODEL_DIR = PROJECT / "models"

TARGET_PATH = (
    PROJECT
    / "data"
    / "target"
    / "mysticup_target_1sec.csv"
)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# LOAD SCALERS
# ==========================================================

time_scaler = joblib.load(
    MODEL_DIR / "time_scaler.pkl"
)

bt_scaler = joblib.load(
    MODEL_DIR / "bt_scaler.pkl"
)

static_scaler = joblib.load(
    MODEL_DIR / "static_scaler.pkl"
)


# ==========================================================
# LOAD DOE
# ==========================================================

corrected_doe = pd.read_csv(DOE_PATH)

corrected_doe["batch_id"] = (
    corrected_doe["ROEST Roast ID"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(int)
)

corrected_doe["csv_file"] = (
    corrected_doe["Profile / Data File"]
    .astype(str)
)


# ==========================================================
# RAW CSV LOADER
# ==========================================================

def load_roast_csv(csv_path):

    df = pd.read_csv(csv_path)

    df.columns = [
        re.sub(
            r"Total cracks(?: \(\d+\))?",
            "Total cracks",
            str(col)
        )
        for col in df.columns
    ]

    return df


# ==========================================================
# FIND RAW CSV
# ==========================================================

def get_csv_path(batch_id):

    batch_id = int(batch_id)

    row = corrected_doe.loc[
        corrected_doe["batch_id"] == batch_id
    ].iloc[0]

    candidate = Path(
        str(row["csv_file"])
    )

    if candidate.exists():
        return candidate

    candidate = RAW_DIR / candidate.name

    if candidate.exists():
        return candidate

    matches = list(
        RAW_DIR.glob(
            f"*{batch_id}*.csv"
        )
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No CSV found for Batch {batch_id}"
        )

    return matches[0]


# ==========================================================
# BUILD SAMPLE
# ==========================================================

def build_sample(batch_id):

    batch_id = int(batch_id)

    row = corrected_doe.loc[
        corrected_doe["batch_id"] == batch_id
    ].iloc[0]

    csv_path = get_csv_path(
        batch_id
    )

    df = load_roast_csv(
        csv_path
    ).copy()

    time_sec = (
        df["Time"]
        .astype(float)
        .to_numpy()
        / 1000.0
    )

    bt = (
        df["Bean temp (°C)"]
        .astype(float)
        .to_numpy()
    )

    X_seq = np.column_stack([
        time_sec[1:],
        bt[:-1]
    ]).astype(np.float32)

    y_seq = (
        bt[1:]
        .astype(np.float32)
    )

    static = np.array([
        row["Planned Airflow (%)"],
        row["Planned Dev Time (s)"],
        row["Actual Preheat Temp (°C)"],
        row["Actual Charge Temp (°C)"]
    ], dtype=np.float32)

    return {
        "batch_id": batch_id,
        "csv_path": str(csv_path),
        "X_seq": X_seq,
        "y_seq": y_seq,
        "static": static
    }


# ==========================================================
# MODEL
# ==========================================================

class TunableHybridLSTMMLP(nn.Module):

    def __init__(
        self,
        lstm_hidden=64,
        mlp_hidden=32,
        fusion_hidden=64,
        dropout=0.2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                4,
                mlp_hidden
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                mlp_hidden,
                mlp_hidden
            ),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                lstm_hidden + mlp_hidden,
                fusion_hidden
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                fusion_hidden,
                1
            )
        )


# ==========================================================
# LOAD FINAL TUNED MODEL
# ==========================================================

checkpoint = torch.load(
    MODEL_DIR
    / "hybrid_random_search_best.pt",
    map_location=device,
    weights_only=False
)

model_config = checkpoint["config"]

model = TunableHybridLSTMMLP(
    lstm_hidden=model_config[
        "lstm_hidden"
    ],
    mlp_hidden=model_config[
        "mlp_hidden"
    ],
    fusion_hidden=model_config[
        "fusion_hidden"
    ],
    dropout=model_config[
        "dropout"
    ]
).to(device)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()


# ==========================================================
# LOAD TARGET PROFILE
# ==========================================================

target_df = pd.read_csv(
    TARGET_PATH
)

target_times = (
    target_df["time_sec"]
    .to_numpy()
)

target_bt = (
    target_df["target_BT_C"]
    .to_numpy()
)

TARGET_END = int(
    target_times.max()
)


# ==========================================================
# PREDICT ROAST WITH SETTINGS
# ==========================================================

def predict_with_settings(
    sample,
    airflow,
    dev_time,
    first_crack_time
):

    candidate_drop = int(
        round(
            first_crack_time
            + dev_time
        )
    )

    candidate_times = np.arange(
        1,
        candidate_drop + 1,
        dtype=np.float32
    )

    time_scaled = (
        time_scaler
        .transform(
            candidate_times.reshape(
                -1,
                1
            )
        )
        .ravel()
        .astype(np.float32)
    )

    static_raw = np.array([
        airflow,
        dev_time,
        sample["static"][2],
        sample["static"][3]
    ], dtype=np.float32)

    static_scaled = (
        static_scaler
        .transform(
            static_raw.reshape(
                1,
                -1
            )
        )
        .astype(np.float32)
    )

    static_tensor = torch.tensor(
        static_scaled,
        dtype=torch.float32,
        device=device
    )

    initial_bt = float(
        sample["X_seq"][0, 1]
    )

    initial_bt_scaled = (
        bt_scaler
        .transform(
            np.array(
                [[initial_bt]]
            )
        )[0, 0]
    )

    prev_bt = torch.tensor(
        [initial_bt_scaled],
        dtype=torch.float32,
        device=device
    )

    predictions = []

    hidden = None

    model.eval()

    with torch.no_grad():

        static_features = (
            model.mlp(
                static_tensor
            )
        )

        for t_scaled in time_scaled:

            time_tensor = torch.tensor(
                [t_scaled],
                dtype=torch.float32,
                device=device
            )

            input_t = torch.stack(
                [
                    time_tensor,
                    prev_bt
                ],
                dim=1
            ).unsqueeze(1)

            lstm_out, hidden = (
                model.lstm(
                    input_t,
                    hidden
                )
            )

            lstm_features = (
                lstm_out.squeeze(1)
            )

            combined = torch.cat(
                [
                    lstm_features,
                    static_features
                ],
                dim=1
            )

            pred_t = (
                model.fusion(
                    combined
                )
                .squeeze(-1)
            )

            predictions.append(
                pred_t.item()
            )

            prev_bt = pred_t

    predictions = np.array(
        predictions
    ).reshape(-1, 1)

    predictions_c = (
        bt_scaler
        .inverse_transform(
            predictions
        )
        .ravel()
    )

    return (
        candidate_times,
        predictions_c,
        candidate_drop
    )


# ==========================================================
# TARGET ERROR
# ==========================================================

def target_error(
    sample,
    airflow,
    dev_time,
    first_crack_time
):

    (
        times,
        predicted_bt,
        drop
    ) = predict_with_settings(
        sample,
        airflow,
        dev_time,
        first_crack_time
    )

    valid = (
        times <= TARGET_END
    )

    compare_times = (
        times[valid]
        .astype(int)
    )

    compare_pred = (
        predicted_bt[valid]
    )

    compare_target = (
        target_bt[
            compare_times
        ]
    )

    mse = mean_squared_error(
        compare_target,
        compare_pred
    )

    rmse = np.sqrt(
        mse
    )

    duration_error = abs(
        drop - TARGET_END
    )

    return (
        mse,
        rmse,
        drop,
        duration_error
    )


# ==========================================================
# PSO OBJECTIVE
# ==========================================================

LAMBDA_CHANGE = 0.01
BETA_DURATION = 0.005


def make_pso_objective(
    sample,
    current_airflow,
    current_dev,
    first_crack_time
):

    initial_bt = float(
        sample["X_seq"][0, 1]
    )

    preheat = float(
        sample["static"][2]
    )

    charge = float(
        sample["static"][3]
    )

    initial_bt_scaled = (
        bt_scaler
        .transform(
            np.array(
                [[initial_bt]]
            )
        )[0, 0]
    )

    def objective(particles):

        n_particles = len(
            particles
        )

        airflow = (
            particles[:, 0]
        )

        dev_time = (
            particles[:, 1]
        )

        candidate_drop = (
            np.rint(
                first_crack_time
                + dev_time
            )
            .astype(int)
        )

        max_drop = int(
            candidate_drop.max()
        )

        static_raw = np.column_stack([
            airflow,
            dev_time,
            np.full(
                n_particles,
                preheat
            ),
            np.full(
                n_particles,
                charge
            )
        ])

        static_scaled = (
            static_scaler
            .transform(
                static_raw
            )
            .astype(np.float32)
        )

        static_tensor = torch.tensor(
            static_scaled,
            dtype=torch.float32,
            device=device
        )

        prev_bt = torch.full(
            (n_particles,),
            float(
                initial_bt_scaled
            ),
            dtype=torch.float32,
            device=device
        )

        times = np.arange(
            1,
            max_drop + 1,
            dtype=np.float32
        )

        scaled_times = (
            time_scaler
            .transform(
                times.reshape(
                    -1,
                    1
                )
            )
            .ravel()
            .astype(np.float32)
        )

        predictions = []

        hidden = None

        model.eval()

        with torch.no_grad():

            static_features = (
                model.mlp(
                    static_tensor
                )
            )

            for t_scaled in scaled_times:

                time_tensor = torch.full(
                    (n_particles,),
                    float(
                        t_scaled
                    ),
                    dtype=torch.float32,
                    device=device
                )

                input_t = torch.stack(
                    [
                        time_tensor,
                        prev_bt
                    ],
                    dim=1
                ).unsqueeze(1)

                lstm_out, hidden = (
                    model.lstm(
                        input_t,
                        hidden
                    )
                )

                lstm_features = (
                    lstm_out.squeeze(1)
                )

                combined = torch.cat(
                    [
                        lstm_features,
                        static_features
                    ],
                    dim=1
                )

                pred_t = (
                    model.fusion(
                        combined
                    )
                    .squeeze(-1)
                )

                predictions.append(
                    pred_t
                )

                prev_bt = pred_t

        predictions = torch.stack(
            predictions,
            dim=1
        )

        predictions_c = (
            predictions
            .cpu()
            .numpy()
            * bt_scaler.scale_[0]
            + bt_scaler.mean_[0]
        )

        mse_values = np.zeros(
            n_particles
        )

        for i in range(
            n_particles
        ):

            compare_end = min(
                candidate_drop[i],
                TARGET_END
            )

            pred_i = (
                predictions_c[
                    i,
                    :compare_end
                ]
            )

            target_i = (
                target_bt[
                    1:
                    compare_end + 1
                ]
            )

            mse_values[i] = np.mean(
                (
                    pred_i
                    - target_i
                ) ** 2
            )

        change_penalty = (
            LAMBDA_CHANGE
            * (
                (
                    airflow
                    - current_airflow
                ) ** 2
                +
                (
                    dev_time
                    - current_dev
                ) ** 2
            )
        )

        duration_error = np.abs(
            candidate_drop
            - TARGET_END
        )

        duration_penalty = (
            BETA_DURATION
            * duration_error ** 2
        )

        return (
            mse_values
            + change_penalty
            + duration_penalty
        )

    return objective


# ==========================================================
# AVAILABLE BATCHES
# ==========================================================

def get_available_batches():

    return sorted(
        corrected_doe[
            "batch_id"
        ]
        .unique()
        .tolist()
    )


# ==========================================================
# FINAL DSS RECOMMENDATION
# ==========================================================

def generate_recommendation(
    batch_id,
    seed=42,
    n_particles=20,
    iterations=30
):

    sample = build_sample(
        batch_id
    )

    df = load_roast_csv(
        sample["csv_path"]
    ).copy()

    df["Time_sec"] = (
        df["Time"] / 1000
    )

    fc_rows = df.loc[
        df["Event"]
        == "First Crack",
        "Time_sec"
    ]

    if len(fc_rows) == 0:

        raise ValueError(
            "No First Crack event found."
        )

    first_crack_time = float(
        fc_rows.iloc[0]
    )

    current_airflow = float(
        sample["static"][0]
    )

    current_dev = float(
        sample["static"][1]
    )

    (
        current_mse,
        current_rmse,
        current_drop,
        _
    ) = target_error(
        sample,
        current_airflow,
        current_dev,
        first_crack_time
    )

    objective = make_pso_objective(
        sample,
        current_airflow,
        current_dev,
        first_crack_time
    )

    np.random.seed(
        seed
    )

    optimizer = (
        ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=2,
            options={
                "c1": 1.5,
                "c2": 1.5,
                "w": 0.7
            },
            bounds=(
                np.array([
                    50.0,
                    30.0
                ]),
                np.array([
                    100.0,
                    90.0
                ])
            )
        )
    )

    start = time.time()

    (
        best_cost,
        best_position
    ) = optimizer.optimize(
        objective,
        iters=iterations,
        verbose=False
    )

    runtime = (
        time.time()
        - start
    )

    recommended_airflow = float(
        best_position[0]
    )

    recommended_dev = float(
        best_position[1]
    )

    (
        recommended_mse,
        recommended_rmse,
        recommended_drop,
        _
    ) = target_error(
        sample,
        recommended_airflow,
        recommended_dev,
        first_crack_time
    )

    raw_improvement = (
        (
            current_mse
            - recommended_mse
        )
        / current_mse
    ) * 100

    if (
        recommended_mse
        < current_mse
    ):

        decision = (
            "ACCEPT RECOMMENDATION"
        )

        final_airflow = (
            recommended_airflow
        )

        final_dev = (
            recommended_dev
        )

        final_mse = (
            recommended_mse
        )

        final_rmse = (
            recommended_rmse
        )

        improvement = (
            raw_improvement
        )

    else:

        decision = (
            "KEEP CURRENT SETTINGS"
        )

        final_airflow = (
            current_airflow
        )

        final_dev = (
            current_dev
        )

        final_mse = (
            current_mse
        )

        final_rmse = (
            current_rmse
        )

        improvement = 0.0

    return {
        "batch_id":
            int(batch_id),

        "first_crack_time_sec":
            first_crack_time,

        "current_airflow_pct":
            current_airflow,

        "current_dev_time_sec":
            current_dev,

        "recommended_airflow_pct":
            recommended_airflow,

        "recommended_dev_time_sec":
            recommended_dev,

        "final_airflow_pct":
            final_airflow,

        "final_dev_time_sec":
            final_dev,

        "current_mse":
            current_mse,

        "final_mse":
            final_mse,

        "current_rmse":
            current_rmse,

        "final_rmse":
            final_rmse,

        "predicted_improvement_pct":
            improvement,

        "decision":
            decision,

        "predicted_drop_sec":
            recommended_drop,

        "optimization_fitness":
            float(best_cost),

        "runtime_sec":
            runtime
    }


# ==========================================================
# PROFILE CURVES FOR GRAPH
# ==========================================================

def get_profile_curves(
    batch_id,
    recommendation
):

    sample = build_sample(
        batch_id
    )

    df = load_roast_csv(
        sample["csv_path"]
    ).copy()

    df["Time_sec"] = (
        df["Time"] / 1000
    )

    first_crack_time = float(
        df.loc[
            df["Event"]
            == "First Crack",
            "Time_sec"
        ].iloc[0]
    )

    (
        current_times,
        current_curve,
        current_drop
    ) = predict_with_settings(
        sample,
        recommendation[
            "current_airflow_pct"
        ],
        recommendation[
            "current_dev_time_sec"
        ],
        first_crack_time
    )

    (
        rec_times,
        rec_curve,
        rec_drop
    ) = predict_with_settings(
        sample,
        recommendation[
            "final_airflow_pct"
        ],
        recommendation[
            "final_dev_time_sec"
        ],
        first_crack_time
    )

    max_time = max(
        TARGET_END,
        current_drop,
        rec_drop
    )

    time_axis = np.arange(
        0,
        max_time + 1
    )

    target_series = np.full(
        max_time + 1,
        np.nan
    )

    current_series = np.full(
        max_time + 1,
        np.nan
    )

    recommended_series = np.full(
        max_time + 1,
        np.nan
    )

    target_series[
        :TARGET_END + 1
    ] = target_bt

    initial_bt = float(
        sample["X_seq"][0, 1]
    )

    current_series[0] = (
        initial_bt
    )

    recommended_series[0] = (
        initial_bt
    )

    current_series[
        1:
        current_drop + 1
    ] = current_curve

    recommended_series[
        1:
        rec_drop + 1
    ] = rec_curve

    curve_df = pd.DataFrame({
        "Time (s)":
            time_axis,

        "Target BT (°C)":
            target_series,

        "Current Predicted BT (°C)":
            current_series,

        "Recommended Predicted BT (°C)":
            recommended_series
    })

    return curve_df

# ==========================================================
# NEW ROAST CSV UPLOAD SUPPORT
# ==========================================================

def build_uploaded_sample(
    uploaded_file,
    airflow,
    dev_time,
    preheat_temp,
    charge_temp
):

    uploaded_file.seek(0)

    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = [
        re.sub(
            r"Total cracks(?: \(\d+\))?",
            "Total cracks",
            str(col)
        )
        for col in df.columns
    ]

    required_columns = [
        "Time",
        "Event",
        "Bean temp (°C)"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    time_sec = (
        df["Time"]
        .astype(float)
        .to_numpy()
        / 1000.0
    )

    bt = (
        df["Bean temp (°C)"]
        .astype(float)
        .to_numpy()
    )

    X_seq = np.column_stack([
        time_sec[1:],
        bt[:-1]
    ]).astype(np.float32)

    y_seq = bt[1:].astype(
        np.float32
    )

    static = np.array([
        airflow,
        dev_time,
        preheat_temp,
        charge_temp
    ], dtype=np.float32)

    sample = {
        "batch_id": "NEW_UPLOAD",
        "X_seq": X_seq,
        "y_seq": y_seq,
        "static": static
    }

    df["Time_sec"] = (
        df["Time"] / 1000
    )

    fc_rows = df.loc[
        df["Event"] == "First Crack",
        "Time_sec"
    ]

    if len(fc_rows) == 0:
        raise ValueError(
            "Uploaded roast has no First Crack event."
        )

    first_crack_time = float(
        fc_rows.iloc[0]
    )

    return (
        sample,
        df,
        first_crack_time
    )


# ==========================================================
# GENERATE RECOMMENDATION FROM UPLOADED CSV
# ==========================================================

def generate_uploaded_recommendation(
    uploaded_file,
    airflow,
    dev_time,
    preheat_temp,
    charge_temp,
    seed=42,
    n_particles=20,
    iterations=30
):

    (
        sample,
        roast_df,
        first_crack_time
    ) = build_uploaded_sample(
        uploaded_file,
        airflow,
        dev_time,
        preheat_temp,
        charge_temp
    )

    current_airflow = float(
        airflow
    )

    current_dev = float(
        dev_time
    )

    (
        current_mse,
        current_rmse,
        current_drop,
        _
    ) = target_error(
        sample,
        current_airflow,
        current_dev,
        first_crack_time
    )

    objective = make_pso_objective(
        sample,
        current_airflow,
        current_dev,
        first_crack_time
    )

    np.random.seed(seed)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2,
        options={
            "c1": 1.5,
            "c2": 1.5,
            "w": 0.7
        },
        bounds=(
            np.array([
                50.0,
                30.0
            ]),
            np.array([
                100.0,
                90.0
            ])
        )
    )

    start = time.time()

    best_cost, best_position = (
        optimizer.optimize(
            objective,
            iters=iterations,
            verbose=False
        )
    )

    runtime = (
        time.time() - start
    )

    recommended_airflow = float(
        best_position[0]
    )

    recommended_dev = float(
        best_position[1]
    )

    (
        recommended_mse,
        recommended_rmse,
        recommended_drop,
        _
    ) = target_error(
        sample,
        recommended_airflow,
        recommended_dev,
        first_crack_time
    )

    improvement = (
        (
            current_mse
            - recommended_mse
        )
        / current_mse
    ) * 100

    if recommended_mse < current_mse:

        decision = (
            "ACCEPT RECOMMENDATION"
        )

        final_airflow = (
            recommended_airflow
        )

        final_dev = (
            recommended_dev
        )

        final_mse = (
            recommended_mse
        )

        final_rmse = (
            recommended_rmse
        )

    else:

        decision = (
            "KEEP CURRENT SETTINGS"
        )

        final_airflow = (
            current_airflow
        )

        final_dev = (
            current_dev
        )

        final_mse = (
            current_mse
        )

        final_rmse = (
            current_rmse
        )

        improvement = 0.0

    result = {
        "batch_id":
            "NEW_UPLOAD",

        "first_crack_time_sec":
            first_crack_time,

        "current_airflow_pct":
            current_airflow,

        "current_dev_time_sec":
            current_dev,

        "recommended_airflow_pct":
            recommended_airflow,

        "recommended_dev_time_sec":
            recommended_dev,

        "final_airflow_pct":
            final_airflow,

        "final_dev_time_sec":
            final_dev,

        "current_mse":
            current_mse,

        "final_mse":
            final_mse,

        "current_rmse":
            current_rmse,

        "final_rmse":
            final_rmse,

        "predicted_improvement_pct":
            improvement,

        "decision":
            decision,

        "predicted_drop_sec":
            recommended_drop,

        "optimization_fitness":
            float(best_cost),

        "runtime_sec":
            runtime
    }

    return (
        result,
        sample,
        first_crack_time
    )


# ==========================================================
# GRAPH FOR UPLOADED ROAST
# ==========================================================

def get_uploaded_profile_curves(
    sample,
    first_crack_time,
    recommendation
):

    (
        current_times,
        current_curve,
        current_drop
    ) = predict_with_settings(
        sample,
        recommendation[
            "current_airflow_pct"
        ],
        recommendation[
            "current_dev_time_sec"
        ],
        first_crack_time
    )

    (
        rec_times,
        rec_curve,
        rec_drop
    ) = predict_with_settings(
        sample,
        recommendation[
            "final_airflow_pct"
        ],
        recommendation[
            "final_dev_time_sec"
        ],
        first_crack_time
    )

    max_time = max(
        TARGET_END,
        current_drop,
        rec_drop
    )

    time_axis = np.arange(
        0,
        max_time + 1
    )

    target_series = np.full(
        max_time + 1,
        np.nan
    )

    current_series = np.full(
        max_time + 1,
        np.nan
    )

    recommended_series = np.full(
        max_time + 1,
        np.nan
    )

    target_series[
        :TARGET_END + 1
    ] = target_bt

    initial_bt = float(
        sample["X_seq"][0, 1]
    )

    current_series[0] = (
        initial_bt
    )

    recommended_series[0] = (
        initial_bt
    )

    current_series[
        1:
        current_drop + 1
    ] = current_curve

    recommended_series[
        1:
        rec_drop + 1
    ] = rec_curve

    return pd.DataFrame({
        "Time (s)":
            time_axis,

        "Target BT (°C)":
            target_series,

        "Current Predicted BT (°C)":
            current_series,

        "Recommended Predicted BT (°C)":
            recommended_series
    })