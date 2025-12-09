import os
import pandas as pd
import streamlit as st
import wandb
from loguru import logger
from dotenv import load_dotenv
import plotly.express as px
import io
import matplotlib.pyplot as plt
from src.utils import plot_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load environment variables from .env file
load_dotenv()

@st.cache_data(ttl=3600)
def get_architecture_name(config):
    """Generates an architecture name based on a run's config."""
    def robust_bool(val):
        if isinstance(val, str):
            return val.lower() in ['true', '1']
        return bool(val)

    use_text = robust_bool(config.get('use_clip_text_features', False))
    use_image = robust_bool(config.get('use_clip_image_features', False))
    use_deepfake = robust_bool(config.get('use_deepfake_detector', False))
    use_encyclopedic = robust_bool(config.get('use_encyclopedic_knowledge', False))

    if use_text and use_image and use_deepfake and use_encyclopedic:
        return 'Baseline'

    parts = []
    if use_text and use_image:
        parts.append('CLIP-Features')
    elif use_text:
        parts.append('CLIP-Text')
    elif use_image:
        parts.append('CLIP-Image')

    if use_deepfake:
        parts.append('DeepfakeDetector')
    if use_encyclopedic:
        parts.append('Encyclopedic')
    
    if not parts:
        return 'No Features'
    return '+'.join(sorted(parts))

@st.cache_data(ttl=3600)
def fetch_all_predictions_for_study(study_name: str, wandb_project: str, wandb_entity: str):
    """Fetches and combines 'test_predictions' tables from all runs in a study."""
    st.info(f"Querying W&B for runs in group: **{study_name}**")
    
    try:
        api = wandb.Api()
        runs = api.runs(
            path=f"{wandb_entity}/{wandb_project}",
            filters={"group": study_name}
        )
    except Exception as e:
        st.error(f"Failed to fetch runs from W&B: {e}")
        return pd.DataFrame()

    st.metric(label="Runs Found in Group", value=len(runs))

    if not runs:
        st.warning("No runs found for the specified study.")
        return pd.DataFrame()

    all_predictions_df = []
    
    progress_bar = st.progress(0, text="Starting data fetch...")
    for i, run in enumerate(runs):
        progress_bar.progress((i + 1) / len(runs), text=f"Fetching data for run: {run.name}")

        architecture = get_architecture_name(run.config)

        predictions_artifact = None
        for artifact in run.logged_artifacts():
            if 'test_predictions' in artifact.name:
                predictions_artifact = artifact
                break
        
        if predictions_artifact:
            try:
                table = predictions_artifact.get("test_predictions")
                df = pd.DataFrame(data=table.data, columns=table.columns)
                df['architecture'] = architecture
                df['run_name'] = run.name
                all_predictions_df.append(df)
            except Exception as e:
                st.warning(f"Could not load 'test_predictions' table from run '{run.name}': {e}")

    progress_bar.empty()

    if not all_predictions_df:
        st.error("No 'test_predictions' tables could be loaded from any run in this study.")
        return pd.DataFrame()

    combined_df = pd.concat(all_predictions_df, ignore_index=True)
    st.success(f"Successfully loaded and combined predictions from {len(all_predictions_df)} runs, totaling {len(combined_df)} samples.")
    return combined_df

def compute_metrics_for_subset(df: pd.DataFrame):
    """Computes and displays metrics for a subset of data, grouped by architecture."""
    if df.empty:
        st.warning("No data to compute metrics on.")
        return

    if 'architecture' not in df.columns:
        st.error("Cannot compute metrics per architecture because 'architecture' column is missing.")
        return

    architectures = df['architecture'].unique()
    
    all_results = []

    neuron_map = {
        '1': 'img_real',
        '2': 'claim_real',
        '3': 'mismatch',
        '4': 'overall'
    }

    for arch in architectures:
        arch_df = df[df['architecture'] == arch]
        if arch_df.empty:
            continue
        
        for i, name in neuron_map.items():
            y_true_col = f'y_true_{name}'
            y_pred_col = f'y_pred_{name}'
            y_prob_col = f'y_prob_{name}'

            if y_true_col not in arch_df.columns or y_pred_col not in arch_df.columns or y_prob_col not in arch_df.columns:
                continue

            y_true = arch_df[y_true_col]
            y_pred = arch_df[y_pred_col]
            y_prob = arch_df[y_prob_col]

            if len(y_true.unique()) <= 1:
                roc_auc = None
            else:
                roc_auc = roc_auc_score(y_true, y_prob)

            metrics = {
                'architecture': arch,
                'Output Neuron': name,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc,
                'support': len(arch_df)
            }
            all_results.append(metrics)

    if not all_results:
        st.warning("Could not compute any metrics for the selected subset.")
        return

    metrics_df = pd.DataFrame(all_results)
    
    id_vars = ['architecture', 'Output Neuron', 'support']
    value_vars = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    value_vars = [v for v in value_vars if v in metrics_df.columns]
    melted_df = metrics_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='metric_name', value_name='metric_value')

    unique_neurons = sorted(melted_df['Output Neuron'].unique())

    for neuron in unique_neurons:
        st.write(f"##### Results for: {neuron}")
        neuron_df = melted_df[melted_df['Output Neuron'] == neuron]
        try:
            pivot_df = neuron_df.pivot_table(
                index='architecture',
                columns='metric_name',
                values='metric_value',
                aggfunc='first'
            )
            support_info = metrics_df[metrics_df['Output Neuron'] == neuron][['architecture', 'support']].drop_duplicates().set_index('architecture')
            pivot_df = pivot_df.merge(support_info, left_index=True, right_index=True)

            cols_order = ['support', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc']
            existing_cols = [col for col in cols_order if col in pivot_df.columns]
            
            st.dataframe(pivot_df[existing_cols].style.format("{:.3f}", na_rep="-"))
        except Exception as e:
            st.warning(f"Could not generate pivot table for {neuron}: {e}")
            st.dataframe(neuron_df)


def main():
    st.set_page_config(layout="wide", page_title="W&B Sample-Level Analysis")

    st.title("Interactive Analysis of W&B Sample Results")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("Setup")
        wandb_project_default = os.getenv("WANDB_PROJECT")
        wandb_entity_default = os.getenv("WANDB_ENTITY")

        study_name = st.text_input("Enter the Optuna Study Name (W&B Group)", value='ablation-fixed-predresMMFAKEBENCH')
        wandb_project = st.text_input("W&B Project", value=wandb_project_default)
        wandb_entity = st.text_input("W&B Entity", value=wandb_entity_default)

        if not (study_name and wandb_project and wandb_entity):
            st.info("Please provide a study name, project, and entity to begin.")
            st.stop()

    # --- Data Fetching ---
    df = fetch_all_predictions_for_study(study_name, wandb_project, wandb_entity)

    if df.empty:
        st.stop()

    st.header("Overall Performance")
    st.write("Metrics calculated on the entire dataset from all runs in the study.")
    compute_metrics_for_subset(df)

    # --- Filtering Section ---
    st.header("Filtered Analysis")
    st.write("Define constraints to analyze a specific subset of the data.")

    filter_cols = st.columns(4)
    with filter_cols[0]:
        arch_options = ['All'] + sorted(df['architecture'].unique().tolist())
        selected_arch = st.selectbox("Filter by `architecture`", options=arch_options)
    with filter_cols[1]:
        fake_cls_options = ['All'] + sorted(df['fake_cls'].unique().tolist())
        selected_fake_cls = st.selectbox("Filter by `fake_cls`", options=fake_cls_options)
    with filter_cols[2]:
        image_source_options = ['All'] + sorted(df['image_source'].unique().tolist())
        selected_image_source = st.selectbox("Filter by `image_source`", options=image_source_options)
    with filter_cols[3]:
        text_source_options = ['All'] + sorted(df['text_source'].unique().tolist())
        selected_text_source = st.selectbox("Filter by `text_source`", options=text_source_options)

    # Apply filters
    filtered_df = df.copy()
    if selected_arch != 'All':
        filtered_df = filtered_df[filtered_df['architecture'] == selected_arch]
    if selected_fake_cls != 'All':
        filtered_df = filtered_df[filtered_df['fake_cls'] == selected_fake_cls]
    if selected_image_source != 'All':
        filtered_df = filtered_df[filtered_df['image_source'] == selected_image_source]
    if selected_text_source != 'All':
        filtered_df = filtered_df[filtered_df['text_source'] == selected_text_source]

    st.metric("Samples in Filtered Subset", len(filtered_df))

    if len(filtered_df) == 0:
        st.warning("No samples match the current filter criteria.")
    elif len(filtered_df) == len(df):
        st.info("Filters match all samples. The metrics below are for the entire dataset.")
    
    st.subheader("Metrics for Filtered Subset")
    compute_metrics_for_subset(filtered_df)

    with st.expander("View Filtered Data"):
        st.dataframe(filtered_df)


    # --- Metric vs. Text Length Analysis ---
    st.header("Metric vs. Text Length Analysis")

    plot_filter_cols = st.columns(4)
    with plot_filter_cols[0]:
        plot_arch_options = ['All'] + sorted(df['architecture'].unique().tolist())
        plot_selected_arch = st.selectbox("Architecture to Plot", options=plot_arch_options, key="plot_arch_select")
    with plot_filter_cols[1]:
        rounding_step = st.slider("Text length rounding step", min_value=10, max_value=200, value=50, step=10, key="rounding_step")
    with plot_filter_cols[2]:
        selected_metric = st.selectbox("Select metric", options=['accuracy', 'precision', 'recall', 'f1'], index=3, key="selected_metric")
    with plot_filter_cols[3]:
        min_support = st.slider("Minimum samples per group", min_value=1, max_value=50, value=5, step=1, key="min_support")

    # Start with the globally filtered dataframe
    plot_df = filtered_df.copy()

    # Apply the specific architecture filter for this plot
    if plot_selected_arch != 'All':
        plot_df = plot_df[plot_df['architecture'] == plot_selected_arch]

    if not plot_df.empty and 'text_raw' in plot_df.columns:
        plot_df['text_raw'] = plot_df['text_raw'].astype(str)
        plot_df['text_len'] = plot_df['text_raw'].str.len()
        plot_df['text_len_rounded'] = (plot_df['text_len'] // rounding_step) * rounding_step

        grouped = plot_df.groupby('text_len_rounded')
        
        plot_data = []
        
        neuron_map = {
            '1': 'img_real',
            '2': 'claim_real',
            '3': 'mismatch',
            '4': 'overall'
        }

        for length_group, group_df in grouped:
            if len(group_df) < min_support:
                continue

            for i, name in neuron_map.items():
                y_true_col = f'y_true_{name}'
                y_pred_col = f'y_pred_{name}'
                
                if y_true_col not in group_df.columns or y_pred_col not in group_df.columns:
                    continue

                y_true = group_df[y_true_col]
                y_pred = group_df[y_pred_col]

                metric_value = 0
                if selected_metric == 'accuracy':
                    metric_value = accuracy_score(y_true, y_pred)
                elif selected_metric == 'precision':
                    metric_value = precision_score(y_true, y_pred, zero_division=0)
                elif selected_metric == 'recall':
                    metric_value = recall_score(y_true, y_pred, zero_division=0)
                elif selected_metric == 'f1':
                    metric_value = f1_score(y_true, y_pred, zero_division=0)
                
                plot_data.append({
                    'text_len_rounded': length_group,
                    'output_neuron': name,
                    'metric_value': metric_value,
                    'support': len(group_df)
                })

        if plot_data:
            final_plot_df = pd.DataFrame(plot_data)
            
            # Create a plot with matplotlib
            fig, ax = plot_utils.get_styled_figure_ax(figsize=(12, 7), aspect='auto')

            # ax.set_title(f"{selected_metric.title()} vs. Text Length for '{plot_selected_arch}' Architecture (Groups < {min_support} samples excluded)")
            ax.set_xlabel(f'Text Length (Binned, step={rounding_step})')
            ax.set_ylabel(selected_metric.title())

            neurons = sorted(final_plot_df['output_neuron'].unique())
            colors = plot_utils.DATASET_COLORS
            
            for i, neuron in enumerate(neurons):
                neuron_df = final_plot_df[final_plot_df['output_neuron'] == neuron]
                # Sort by text length to ensure lines are drawn correctly
                neuron_df = neuron_df.sort_values('text_len_rounded')
                ax.plot(neuron_df['text_len_rounded'], neuron_df['metric_value'], 
                        marker='o', linestyle='-', label=neuron, color=colors[i % len(colors)])

            # Create a legend
            if neurons:
                plot_utils.style_legend(ax, ncol=len(neurons), bbox_to_anchor=(0.5, 1.15))

            st.pyplot(fig)

            # Add download button
            save_path = "reports/figures/test/application/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            file_name = f"{selected_metric}_vs_text_length_{plot_selected_arch.replace('+', '_')}_{study_name}.svg"
            
            svg_buffer = io.BytesIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches='tight')
            svg_buffer.seek(0)

            st.download_button(
                label="Download plot as SVG",
                data=svg_buffer,
                file_name=file_name,
                mime="image/svg+xml"
            )
        else:
            st.warning("No data groups meet the criteria (architecture, support, etc.) to generate the plot.")
    else:
        st.info("No data to plot. Adjust the main filters or the plot-specific architecture filter.")


if __name__ == "__main__":
    main()
