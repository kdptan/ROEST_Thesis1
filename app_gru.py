from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dss_backend_gru import (
    generate_recommendation,
    get_available_batches,
    get_profile_curves,
    generate_uploaded_recommendation,
    get_uploaded_profile_curves,
    device
)


# ==========================================================
# PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="ROEST Coffee Roasting DSS",
    page_icon="☕",
    layout="wide"
)

PROJECT = Path(__file__).resolve().parent

RESULTS_DIR = (
    PROJECT / "results"
)


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "ROEST Coffee Roasting Decision Support System"
)

st.caption(
    "Hybrid GRU-MLP + Particle Swarm Optimization"
)

st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(
    "Roast Input"
)

input_mode = st.sidebar.radio(
    "Choose input mode",
    [
        "Historical Batch",
        "Upload New Roast CSV"
    ]
)

st.sidebar.write(
    f"Model device: **{str(device).upper()}**"
)

st.sidebar.info(
    "The system analyzes a completed roast "
    "and generates recommended settings "
    "for the next roast."
)


# ==========================================================
# HISTORICAL MODE
# ==========================================================

if input_mode == "Historical Batch":

    available_batches = (
        get_available_batches()
    )

    selected_batch = (
        st.sidebar.selectbox(
            "Select completed roast batch",
            available_batches
        )
    )

    st.subheader(
        "Completed Roast Analysis"
    )

    st.write(
        f"Selected Batch: **{selected_batch}**"
    )

    run_button = st.button(
        "Generate Next-Roast Recommendation",
        type="primary",
        use_container_width=True
    )

    if run_button:

        with st.spinner(
            "Analyzing roast and running "
            "Particle Swarm Optimization..."
        ):

            try:

                result = (
                    generate_recommendation(
                        selected_batch
                    )
                )

                st.session_state[
                    "dss_result"
                ] = result

                st.session_state[
                    "result_mode"
                ] = "historical"

                output_path = (
                    RESULTS_DIR
                    / (
                        f"gru_dss_recommendation_"
                        f"batch_{selected_batch}_final.csv"
                    )
                )

                pd.DataFrame(
                    [result]
                ).to_csv(
                    output_path,
                    index=False
                )

                st.session_state[
                    "output_path"
                ] = str(
                    output_path
                )

            except Exception as e:

                st.error(
                    f"Recommendation failed: {e}"
                )


# ==========================================================
# UPLOAD MODE
# ==========================================================

else:

    st.subheader(
        "Upload New Completed Roast"
    )

    uploaded_file = st.file_uploader(
        "Upload ROEST roast CSV",
        type=["csv"]
    )

    st.markdown(
        "### Completed Roast Settings"
    )

    st.caption(
        "Enter the settings used for the uploaded completed roast."
    )

    col1, col2 = st.columns(2)

    with col1:

        upload_airflow = st.number_input(
            "Planned Airflow (%)",
            min_value=50.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        upload_preheat = st.number_input(
            "Actual Preheat Temperature (°C)",
            min_value=150.0,
            max_value=220.0,
            value=180.0,
            step=0.1
        )

    with col2:

        upload_dev = st.number_input(
            "Planned Development Time (s)",
            min_value=30.0,
            max_value=90.0,
            value=60.0,
            step=1.0
        )

        upload_charge = st.number_input(
            "Actual Charge Temperature (°C)",
            min_value=150.0,
            max_value=220.0,
            value=175.0,
            step=0.1
        )

    if uploaded_file is not None:

        st.success(
            f"Loaded: {uploaded_file.name}"
        )

    run_upload_button = st.button(
        "Analyze Uploaded Roast",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )

    if run_upload_button:

        with st.spinner(
            "Analyzing uploaded roast and running "
            "Particle Swarm Optimization..."
        ):

            try:

                (
                    result,
                    uploaded_sample,
                    uploaded_fc
                ) = generate_uploaded_recommendation(
                    uploaded_file,
                    upload_airflow,
                    upload_dev,
                    upload_preheat,
                    upload_charge
                )

                st.session_state[
                    "dss_result"
                ] = result

                st.session_state[
                    "result_mode"
                ] = "upload"

                st.session_state[
                    "uploaded_sample"
                ] = uploaded_sample

                st.session_state[
                    "uploaded_fc"
                ] = uploaded_fc

                output_path = (
                    RESULTS_DIR
                    / "gru_dss_recommendation_uploaded_roast_final.csv"
                )

                pd.DataFrame(
                    [result]
                ).to_csv(
                    output_path,
                    index=False
                )

                st.session_state[
                    "output_path"
                ] = str(
                    output_path
                )

            except Exception as e:

                st.error(
                    f"Uploaded roast analysis failed: {e}"
                )


# ==========================================================
# DISPLAY RESULT
# ==========================================================

show_result = False

if "dss_result" in st.session_state:

    if (
        input_mode == "Historical Batch"
        and st.session_state.get(
            "result_mode"
        ) == "historical"
    ):
        show_result = True

    elif (
        input_mode == "Upload New Roast CSV"
        and st.session_state.get(
            "result_mode"
        ) == "upload"
    ):
        show_result = True


if show_result:

    result = (
        st.session_state[
            "dss_result"
        ]
    )

    st.divider()

    if (
        st.session_state[
            "result_mode"
        ] == "historical"
    ):

        st.header(
            f"Recommendation for Batch "
            f"{result['batch_id']}"
        )

    else:

        st.header(
            "Recommendation for Uploaded Roast"
        )


    # ======================================================
    # DECISION
    # ======================================================

    if (
        result["decision"]
        == "ACCEPT RECOMMENDATION"
    ):

        st.success(
            "Decision: ACCEPT RECOMMENDATION"
        )

    else:

        st.warning(
            "Decision: KEEP CURRENT SETTINGS"
        )


    # ======================================================
    # SETTINGS COMPARISON
    # ======================================================

    st.subheader(
        "Current vs Recommended Settings"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Current Roast"
        )

        st.metric(
            "Airflow",
            f"{result['current_airflow_pct']:.2f}%"
        )

        st.metric(
            "Development Time",
            f"{result['current_dev_time_sec']:.2f} s"
        )

    with col2:

        st.markdown(
            "### Recommended Next Roast"
        )

        airflow_change = (
            result["final_airflow_pct"]
            - result["current_airflow_pct"]
        )

        dev_change = (
            result["final_dev_time_sec"]
            - result["current_dev_time_sec"]
        )

        st.metric(
            "Airflow",
            f"{result['final_airflow_pct']:.2f}%",
            delta=f"{airflow_change:+.2f}%",
            delta_color="off"
        )

        st.metric(
            "Development Time",
            f"{result['final_dev_time_sec']:.2f} s",
            delta=f"{dev_change:+.2f} s",
            delta_color="off"
        )


    # ======================================================
    # PERFORMANCE
    # ======================================================

    st.subheader(
        "Predicted Profile Performance"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Current RMSE",
            f"{result['current_rmse']:.4f} °C"
        )

    with m2:

        st.metric(
            "Recommended RMSE",
            f"{result['final_rmse']:.4f} °C"
        )

    with m3:

        st.metric(
            "Predicted Improvement",
            f"{result['predicted_improvement_pct']:.2f}%"
        )


    # ======================================================
    # PROFILE GRAPH
    # ======================================================

    st.subheader(
        "Bean Temperature Profile Comparison"
    )

    if (
        st.session_state[
            "result_mode"
        ] == "historical"
    ):

        curve_df = get_profile_curves(
            result["batch_id"],
            result
        )

    else:

        curve_df = get_uploaded_profile_curves(
            st.session_state[
                "uploaded_sample"
            ],
            st.session_state[
                "uploaded_fc"
            ],
            result
        )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        curve_df["Time (s)"],
        curve_df["Target BT (°C)"],
        label="Target BT",
        linewidth=2.5
    )

    ax.plot(
        curve_df["Time (s)"],
        curve_df[
            "Current Predicted BT (°C)"
        ],
        label="Current Predicted BT",
        linewidth=2
    )

    ax.plot(
        curve_df["Time (s)"],
        curve_df[
            "Recommended Predicted BT (°C)"
        ],
        label="Recommended Predicted BT",
        linewidth=2
    )

    ax.set_xlabel(
        "Roast Time (seconds)"
    )

    ax.set_ylabel(
        "Bean Temperature (°C)"
    )

    ax.set_title(
        "Target vs Current vs Recommended "
        "Bean Temperature Profile"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.caption(
        "Target Bean Temperature profile compared "
        "with the current predicted roast and "
        "recommended predicted roast."
    )


    # ======================================================
    # ROAST INFORMATION
    # ======================================================

    st.subheader(
        "Roast Information"
    )

    info1, info2, info3 = (
        st.columns(3)
    )

    with info1:

        st.metric(
            "First Crack",
            f"{result['first_crack_time_sec']:.0f} s"
        )

    with info2:

        st.metric(
            "Predicted Drop",
            f"{result['predicted_drop_sec']:.0f} s"
        )

    with info3:

        st.metric(
            "Optimization Runtime",
            f"{result['runtime_sec']:.2f} s"
        )


    # ======================================================
    # NEXT ROAST PLAN
    # ======================================================

    st.subheader(
        "Next-Roast Plan"
    )

    if (
        result["decision"]
        == "ACCEPT RECOMMENDATION"
    ):

        st.write(
            f"Set airflow to "
            f"**{result['final_airflow_pct']:.2f}%** "
            f"and plan a development duration of "
            f"**{result['final_dev_time_sec']:.2f} seconds**."
        )

        st.write(
            f"The model predicts approximately "
            f"**{result['predicted_improvement_pct']:.2f}%** "
            f"lower Bean Temperature profile error "
            f"relative to the selected target profile."
        )

    else:

        st.write(
            "The optimized candidate did not improve "
            "the predicted Bean Temperature profile."
        )

        st.write(
            "The system recommends keeping "
            "the current roast settings."
        )


    # ======================================================
    # SAVED OUTPUT
    # ======================================================

    if (
        "output_path"
        in st.session_state
    ):

        st.caption(
            "Recommendation saved to: "
            + st.session_state[
                "output_path"
            ]
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Decision-support only. Final roast adjustments "
    "remain under the control of the coffee roaster."
)