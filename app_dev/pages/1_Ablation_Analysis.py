import os
import pandas as pd
import streamlit as st
import wandb
import typer
from loguru import logger
from pandas import json_normalize
from dotenv import load_dotenv
import plotly.express as px
import io
import matplotlib.pyplot as plt
from src.utils import plot_utils

# Load environment variables from .env file
load_dotenv()

def fetch_wandb_data(study_name: str, wandb_project: str, wandb_entity: str) -> pd.DataFrame:
    """Fetches run data from W&B for a given study."""
    group_name = f"{study_name}"
    st.info(f"Querying W&B for runs in group: **{group_name}**")

    logger.info(f"Fetching data for study '{study_name}' from project '{wandb_entity}/{wandb_project}' with group filter '{group_name}'")
    try:
        api = wandb.Api()
        runs = api.runs(
            path=f"{wandb_entity}/{wandb_project}",
            filters={"group": group_name}
        )

        st.metric(label="Runs Returned by W&B API", value=len(runs))

        if not runs:
            st.warning("No runs found for the specified study and group.")
            return pd.DataFrame()

        summary_list = []
        for run in runs:
            summary = run.summary._json_dict
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            summary.update(config)
            summary_list.append(summary)

        df = pd.DataFrame(summary_list)

        if 'test_metrics' in df.columns:
            # Flatten the nested metric dictionaries
            metrics_flat = json_normalize(df['test_metrics'].dropna(), sep='_')
            df = pd.concat([df.drop(columns=['test_metrics']), metrics_flat], axis=1)

        df.rename(columns={
            'clip_use_text_features': 'use_clip_text_features',
            'clip_use_image_features': 'use_clip_image_features'
        }, inplace=True)

        # If force_clip_choice is present, use it to correct use_clip_image_features and use_clip_text_features
        if 'force_clip_choice' in df.columns:
            st.info("Found 'force_clip_choice'. Applying correction to 'use_clip_image_features' and 'use_clip_text_features'.")
            df.loc[df['force_clip_choice'] < 0.5, 'use_clip_image_features'] = True
            df.loc[df['force_clip_choice'] > 0.5, 'use_clip_text_features'] = True

        return df

    except Exception as e:
        st.error(f"Failed to fetch data from W&B: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(layout="wide", page_title="W&B Study Analysis")

    st.title("Interactive Analysis of W&B Study Results")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("Setup")
        wandb_project_default = os.getenv("WANDB_PROJECT")
        wandb_entity_default = os.getenv("WANDB_ENTITY")

        study_name = st.text_input("Enter the Optuna Study Name (W&B Group)", value='ablation_2025-10-25-11-23_test')
        wandb_project = st.text_input("W&B Project", value=wandb_project_default)
        wandb_entity = st.text_input("W&B Entity", value=wandb_entity_default)

        if not (study_name and wandb_project and wandb_entity):
            st.info("Please provide a study name, project, and entity to begin.")
            st.stop()

    # --- Data Fetching and Display ---
    df = fetch_wandb_data(study_name, wandb_project, wandb_entity)

    if df.empty:
        st.stop()

    # Pre-process columns to handle unhashable types like dicts
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # The set operation will fail if the column contains unhashable types
                set(df[col].dropna())
            except TypeError:
                # If it fails, convert the column to string
                df[col] = df[col].astype(str)

    # Clean up DataFrame
    cols_to_drop = [col for col in df.columns if col.startswith(('_', 'wandb'))]
    cols_to_drop.extend(['name', 'dataset', 'epoch', 'step', 'loss'])
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')

    # Remove columns that have only one unique value (i.e., constant columns)
    constant_cols_to_drop = [
        col for col in df_cleaned.columns
        if df_cleaned[col].nunique() == 1
    ]
    if constant_cols_to_drop:
        df_cleaned = df_cleaned.drop(columns=constant_cols_to_drop)
        st.info(f"Note: The following columns were excluded from the analysis because they contain only one unique value: {', '.join(constant_cols_to_drop)}")

    with st.expander("Raw Data"):
        st.dataframe(df)
    with st.expander("Cleaned Data for Analysis"):
        st.metric(label="Number of Runs (Lines in Plot)", value=len(df_cleaned))
        st.dataframe(df_cleaned)

    # --- Analysis Section ---
    st.header("Analysis and Visualizations")

    # --- Architecture Performance ---
    st.subheader("Architecture Performance")
    st.write("This section analyzes the performance of different model architectures based on the features they use.")

    # Define and create the architecture column
    def get_architecture_name(row):
        use_text = bool(row.get('use_clip_text_features', False))
        use_image = bool(row.get('use_clip_image_features', False))
        use_deepfake = bool(row.get('use_deepfake_detector', False))
        use_encyclopedic = bool(row.get('use_encyclopedic_knowledge', False))

        # Check for the specific condition: all 'use_' features are True
        if use_text and use_image and use_deepfake and use_encyclopedic:
            # return 'Baseline' 
            return 'Global'


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
            return 'No Features' # Changed from 'Baseline' to avoid conflict
        return '+'.join(sorted(parts))

    arch_cols = ['use_clip_text_features', 'use_clip_image_features', 'use_deepfake_detector', 'use_encyclopedic_knowledge']
    
    # Check if all necessary columns are present in the dataframe
    if all(c in df_cleaned.columns for c in arch_cols):
        df_cleaned['architecture'] = df_cleaned.apply(get_architecture_name, axis=1)

        # Identify metrics for the architecture plot
        all_numeric_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype in ['float64', 'int64']]
        output_metric_cols = [m for m in all_numeric_cols if m.startswith('output_')]

        if output_metric_cols:
            # Melt the dataframe to handle metrics properly
            id_vars_arch = [c for c in df_cleaned.columns if c not in output_metric_cols]
            df_arch_melted = df_cleaned.melt(
                id_vars=id_vars_arch,
                value_vars=output_metric_cols,
                var_name='metric_full',
                value_name='metric_value'
            )
            
            # Extract metric name and output neuron
            extracted_data = df_arch_melted['metric_full'].str.extract(r'output_(\d+)_(.*)')
            if extracted_data.shape[1] == 2:
                neuron_map = {
                    '1': 'img_real',
                    '2': 'claim_real',
                    '3': 'mismatch',
                    '4': 'overall_truth'
                }
                df_arch_melted['output_neuron'] = extracted_data[0].map(neuron_map).fillna("Output " + extracted_data[0])
                df_arch_melted['metric_name'] = extracted_data[1]

                # Aggregate data for plotting (mean and std)
                df_agg = df_arch_melted.groupby(['architecture', 'metric_name', 'output_neuron'])['metric_value'].agg(['mean', 'std', 'count']).reset_index()

                # --- Results Table ---
                with st.expander("View Tabulated Results (Mean Â± Std)"):
                    def format_result_string(row):
                        if row['count'] > 1:
                            return f"{row['mean']:.3f} Â± {row['std']:.3f}"
                        else:
                            return f"{row['mean']:.3f}"
                    df_agg['result_str'] = df_agg.apply(format_result_string, axis=1)
                    unique_neurons = sorted(df_agg['output_neuron'].unique())

                    for neuron in unique_neurons:
                        st.write(f"#### Results for: {neuron}")
                        neuron_df = df_agg[df_agg['output_neuron'] == neuron]
                        try:
                            pivot_df = neuron_df.pivot_table(
                                index='architecture',
                                columns='metric_name',
                                values='result_str',
                                aggfunc='first'
                            )
                            st.dataframe(pivot_df)
                        except Exception as e:
                            st.warning(f"Could not generate pivot table for {neuron}: {e}")
                            st.dataframe(neuron_df)

                    st.write("---")
                    st.write("#### Combined Table")
                    try:
                        pivot_df_combined = df_agg.pivot_table(
                            index=['architecture', 'output_neuron'],
                            columns='metric_name',
                            values='result_str',
                            aggfunc='first'
                        ).reset_index()
                        st.dataframe(pivot_df_combined)
                    except Exception as e:
                        st.warning(f"Could not generate combined pivot table: {e}")
                

                # --- Run Distribution and Divergence ---
                st.subheader("Run Distribution and Divergence")
                st.write("This section shows how many runs exist for each architecture and how their hyperparameters vary. The best model for the selected architecture (based on the chosen metric) will be highlighted.")

                arch_counts = df_cleaned['architecture'].value_counts()
                st.bar_chart(arch_counts, use_container_width=True)

                selected_arch = st.selectbox("Select an architecture to inspect its hyperparameter variance:", options=arch_counts.index)
                # --- Best Metric Selection ---
                best_metric = st.selectbox(
                    "Select metric to determine the 'best' model for divergence analysis:",
                    options=sorted(output_metric_cols),
                    index=0
                )
                if not best_metric:
                    st.warning("Please select a metric to determine the best model.")
                    st.stop()

                if selected_arch:
                    arch_df = df_cleaned[df_cleaned['architecture'] == selected_arch]
                    
                    if len(arch_df) > 1:
                        best_run_for_arch = arch_df.loc[arch_df[best_metric].idxmax()]

                        def is_metric(col_name):
                            metric_keywords = ['acc', 'f1', 'loss', 'precision', 'recall', 'output_', 'metric_']
                            return any(keyword in col_name.lower() for keyword in metric_keywords)

                        all_cols = arch_df.columns
                        potential_metrics = {col for col in all_cols if is_metric(col)}
                        hyperparameters = [col for col in all_cols if col not in potential_metrics and col != 'architecture']
                        
                        varying_cols = [
                            col for col in hyperparameters 
                            if arch_df[col].nunique() > 1 and not col in 'trial_number'
                        ]

                        if not varying_cols:
                            st.info("All runs for this architecture have identical hyperparameters.")
                        else:
                            st.write(f"The following hyperparameters vary for the **{selected_arch}** architecture:")
                            
                            # Use Streamlit columns for layout
                            cols = st.columns(3)
                            col_idx = 0

                            for param in varying_cols:
                                with cols[col_idx]:
                                    st.write(f"**Best model's {param}:** `{best_run_for_arch[param]}`")
                                    if arch_df[param].dtype in ['float64', 'int64']: # Numerical -> histogram
                                        fig = px.histogram(arch_df, x=param, title=f"Distribution of {param}", nbins=20)
                                        fig.add_vline(x=best_run_for_arch[param], line_dash="dash", line_color="red", annotation_text="Best Model")
                                        st.plotly_chart(fig, use_container_width=True, key=f"hist_{selected_arch}_{param}")
                                    else: # Categorical -> bar plot
                                        param_counts = arch_df[param].value_counts().reset_index(name='count') # Specify name for count column
                                        param_counts.columns = ['value', 'count']
                                        fig = px.bar(param_counts, x='value', y='count', title=f"Distribution of {param}")
                                        
                                        best_value = best_run_for_arch[param]
                                        if best_value in param_counts['value'].values:
                                            colors = ['blue'] * len(param_counts)
                                            best_value_idx = param_counts[param_counts['value'] == best_value].index[0]
                                            colors[best_value_idx] = 'red'
                                            fig.update_traces(marker_color=colors)

                                        st.plotly_chart(fig, use_container_width=True, key=f"bar_cat_{selected_arch}_{param}")
                                
                                col_idx = (col_idx + 1) % 3
                    else:
                        st.info("Only one run found for this architecture.")


                base_metrics = sorted(df_agg['metric_name'].unique())

                bar_tab, line_tab = st.tabs([":bar_chart: Bar Plots", ":chart_with_upwards_trend: Line Plots"])

                with bar_tab:
                    fig = None # Reset fig
                    for base_metric in base_metrics:
                        st.write(f"#### {base_metric.replace('_', ' ').title()}")
                        plot_data = df_agg[df_agg['metric_name'] == base_metric]
                        
                        fig = px.bar(
                            plot_data,
                            x='architecture',
                            y='mean',
                            error_y='std',
                            color='output_neuron',
                            barmode='group',
                            title=f"Architecture Performance for {base_metric.replace('_', ' ').title()}",
                            labels={'mean': base_metric.replace('_', ' ').title(), 'architecture': 'Architecture', 'output_neuron': 'Output Neuron'}
                        )
                        if not plot_data.empty:
                            y_min = plot_data['mean'].min() - 0.1
                            y_max = plot_data['mean'].max() + 0.1
                            fig.update_yaxes(range=[max(0, y_min), min(1.05, y_max)])
                        st.plotly_chart(fig, use_container_width=True, key=f"bar_{base_metric}")

                with line_tab:
                    fig = None # Reset fig
                    for base_metric in base_metrics:
                        st.write(f"#### {base_metric.replace('_', ' ').title()}")
                        plot_data = df_agg[df_agg['metric_name'] == base_metric]

                        fig = px.line(
                            plot_data,
                            x='architecture',
                            y='mean',
                            color='output_neuron',
                            markers=True,
                            title=f"Architecture Performance for {base_metric.replace('_', ' ').title()}",
                            labels={'mean': base_metric.replace('_', ' ').title(), 'architecture': 'Architecture', 'output_neuron': 'Output Neuron'}
                        )
                        if not plot_data.empty:
                            y_min = plot_data['mean'].min() - 0.1
                            y_max = plot_data['mean'].max() + 0.1
                            fig.update_yaxes(range=[max(0, y_min), min(1.05, y_max)])
                        st.plotly_chart(fig, use_container_width=True, key=f"line_{base_metric}")
        else:
            st.info("No 'output_*' metrics found to generate architecture performance plots.")
    else:
        st.warning(f"Could not create architecture plot. One or more required columns are missing: {', '.join(arch_cols)}")


    # Identify column types
    metrics = st.multiselect(
        "Select Metrics to Analyze",
        options=sorted([col for col in df_cleaned.columns if df_cleaned[col].dtype in ['float64', 'int64']]),
        default=sorted([col for col in df_cleaned.columns if df_cleaned[col].dtype in ['float64', 'int64'] and ('acc' in col or 'f1' in col)])
    )

    # Explicitly convert boolean columns to integers (0 or 1) for wider compatibility
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'bool':
            df_cleaned[col] = df_cleaned[col].astype(int)

    hyperparameters = [col for col in df_cleaned.columns if col not in metrics]
    categorical_hyperparams = [col for col in hyperparameters if df_cleaned[col].nunique() < 10]
    numerical_hyperparams = [col for col in hyperparameters if col not in categorical_hyperparams and df_cleaned[col].dtype in ['float64', 'int64']]

    if not metrics:
        st.warning("Please select at least one metric to analyze.")
        st.stop()

    # --- Best Model ---
    st.subheader("Best Model Analysis")
    best_metric = st.selectbox("Select metric to determine the 'best' model", options=metrics)
    if best_metric:
        best_run = df_cleaned.loc[df_cleaned[best_metric].idxmax()]
        st.write(f"The best model based on **{best_metric}** is:")
        st.json(best_run.to_json())


    # --- Correlation Matrix ---
    st.subheader("Correlation Matrix")
    st.write("This matrix shows the correlation between model parameters and all numerical metrics.")
    
    # Identify all numerical-like columns
    all_numeric_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype in ['float64', 'int64']]
    # Identify output metrics
    output_metric_cols = [m for m in all_numeric_cols if m.startswith('output_')]
    
    # Select all numerical-like columns for the correlation matrix.
    # This includes metrics, numerical hyperparameters, and boolean hyperparameters (which were converted to int).
    numerical_cols_for_corr = sorted([
        col for col in df_cleaned.columns 
        if df_cleaned[col].dtype in ['float64', 'int64'] and col not in ('force_clip_choice')
    ])
    
    if numerical_cols_for_corr:
        correlation_matrix = df_cleaned[numerical_cols_for_corr].corr()
        
        # Filter rows to only include non-output parameters
        model_param_cols = [col for col in numerical_cols_for_corr if col not in output_metric_cols]
        
        # Ensure that there are model parameters to display
        if model_param_cols:
            correlation_matrix_filtered = correlation_matrix.loc[model_param_cols]
            
            fig_corr = px.imshow(
                correlation_matrix_filtered, # Use the filtered matrix
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                title="Correlation Matrix (Model Parameters vs. All Metrics)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No model parameters found to display in the correlation matrix.")
    else:
        st.info("No numerical columns found to generate a correlation matrix.")


    # --- Parallel Coordinates ---
    st.subheader("Hyperparameter Importance (Parallel Coordinates)")

    # All hyperparameters (numerical and categorical) and metrics should be available for selection
    all_dims = sorted(hyperparameters + metrics)

    # --- START of new filtering logic ---
    st.write("---")
    st.write("#### Define Constraints for Parallel Plot")

    if 'parallel_plot_filters' not in st.session_state:
        st.session_state.parallel_plot_filters = []

    def add_filter_callback():
        # Add a new filter configuration to the session state
        if all_dims:
            # Check if there are any unique values for the default column
            if len(df_cleaned[all_dims[0]].unique()) > 0:
                st.session_state.parallel_plot_filters.append(
                    {'column': all_dims[0], 'value': df_cleaned[all_dims[0]].unique()[0]}
                )
            else:
                st.warning(f"Cannot add filter for '{all_dims[0]}' as it has no unique values.")

    def remove_filter_callback(index):
        # Remove a filter from the session state by its index
        if 0 <= index < len(st.session_state.parallel_plot_filters):
            st.session_state.parallel_plot_filters.pop(index)

    # Display existing filters and allow modification/removal
    for i, p_filter in enumerate(st.session_state.parallel_plot_filters):
        cols = st.columns([3, 3, 1])
        with cols[0]:
            # Select column for filtering
            current_col_index = all_dims.index(p_filter['column']) if p_filter['column'] in all_dims else 0
            p_filter['column'] = st.selectbox(f"Column to filter", all_dims, index=current_col_index, key=f"filter_col_{i}")
        
        with cols[1]:
            # Select value for filtering based on the chosen column
            unique_values = df_cleaned[p_filter['column']].unique()
            if len(unique_values) > 0:
                # Convert to list to use .index()
                unique_values_list = list(unique_values)
                current_val_index = unique_values_list.index(p_filter['value']) if p_filter['value'] in unique_values_list else 0
                p_filter['value'] = st.selectbox(f"Value to match", unique_values, index=current_val_index, key=f"filter_val_{i}")
            else:
                st.text(f"No values to select for '{p_filter['column']}'.")


        with cols[2]:
            # "Remove" button
            st.button("X", on_click=remove_filter_callback, args=(i,), key=f"remove_filter_{i}")

    st.button("Add Constraint", on_click=add_filter_callback)

    # Apply the defined filters to the dataframe
    parallel_plot_df = df_cleaned.copy()
    if st.session_state.parallel_plot_filters:
        query_parts = []
        for p_filter in st.session_state.parallel_plot_filters:
            col = p_filter['column']
            val = p_filter['value']
            # Safely construct a query part
            if isinstance(val, str):
                query_parts.append(f"`{col}` == '{val}'")
            else:
                query_parts.append(f"`{col}` == {val}")
        
        if query_parts:
            query_str = " and ".join(query_parts)
            try:
                parallel_plot_df = parallel_plot_df.query(query_str)
                st.info(f"Data filtered to {len(parallel_plot_df)} rows based on constraints.")
            except Exception as e:
                st.error(f"Failed to apply filters: {e}")
    # --- END of new filtering logic ---

    if all_dims:
        # Widget to select the main output/color dimension
        # Default to the last metric in the sorted list if available
        default_color_name = 'output_4_accuracy'
        default_color_index = all_dims.index(default_color_name) if default_color_name in all_dims else len(all_dims) - 1
        color_dim = st.selectbox(
            "Select Output Dimension (will be the last axis and drive color)",
            options=all_dims,
            index=default_color_index
        )

        # Widget to select all other dimensions for the axes
        default_dims = ['use_deepfake_detector',
                        'use_clip_text_features', 'use_clip_image_features', 'use_encyclopedic_knowledge']
        selected_dims = st.multiselect(
            "Select Dimensions for Parallel Plot Axes",
            options=all_dims,
            default=default_dims
        )

        if selected_dims and color_dim:
            # Ensure the color dimension is always the last axis
            plot_dims = [d for d in selected_dims if d != color_dim] + [color_dim]

            # Parallel coordinates plots cannot render rows with any missing values in the selected dimensions.
            # We filter them out and warn the user.
            plot_df = parallel_plot_df[plot_dims]
            initial_rows = len(plot_df)
            plot_df_cleaned = plot_df.dropna()
            dropped_rows = initial_rows - len(plot_df_cleaned)

            if dropped_rows > 0:
                st.warning(f"**{dropped_rows} runs were hidden from the parallel plot** because they contained missing values for one or more of the selected axes. Unselect axes to include more runs.")

            if plot_df_cleaned.empty:
                st.error("No data remains after removing runs with missing values. Please select fewer axes.")
            else:
                fig = px.parallel_coordinates(
                    plot_df_cleaned,
                    dimensions=plot_dims,
                    color=color_dim,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Parallel Coordinates Plot of Hyperparameters and Metrics",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one axis dimension and an output dimension.")

    # --- Metric vs. Text Length Analysis ---
    st.subheader("Metric vs. Text Length Analysis")
    st.write("This section allows you to create a scatter plot of a metric against a numerical hyperparameter (like text length) using the constrained data from the parallel plot above, and download it.")

    # Use the same numerical_hyperparams and metrics as defined before
    if numerical_hyperparams and metrics:
        col1, col2 = st.columns(2)
        with col1:
            x_axis_param = st.selectbox(
                "Select X-axis (Numerical Hyperparameter)",
                options=sorted(numerical_hyperparams),
                key="text_length_x_axis"
            )
        with col2:
            y_axis_metric = st.selectbox(
                "Select Y-axis (Metric)",
                options=sorted(metrics),
                key="text_length_y_axis"
            )

        if x_axis_param and y_axis_metric:
            st.write(f"Plotting **{y_axis_metric}** vs. **{x_axis_param}** for the **{len(parallel_plot_df)}** runs matching the constraints.")
            # Create plot
            fig, ax = plot_utils.get_styled_figure_ax(figsize=(10, 7), aspect='auto')
            
            plot_df = parallel_plot_df[[x_axis_param, y_axis_metric]].dropna()

            if not plot_df.empty:
                ax.scatter(plot_df[x_axis_param], plot_df[y_axis_metric],
                           s=90,
                           color=plot_utils.DATASET_COLORS[0],
                           edgecolors=plot_utils.DATASET_LINE_COLORS[0],
                           linewidths=1.5,
                           zorder=3)

                ax.set_xlabel(x_axis_param)
                ax.set_ylabel(y_axis_metric)
                ax.set_title(f"{y_axis_metric} vs. {x_axis_param}")
                
                # Adjust limits without forcing equal aspect
                x_min, x_max = plot_df[x_axis_param].min(), plot_df[x_axis_param].max()
                y_min, y_max = plot_df[y_axis_metric].min(), plot_df[y_axis_metric].max()
                x_pad = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 0.1
                y_pad = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

                st.pyplot(fig)

                # Save plot button
                save_path = "reports/figures/test/application/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                file_name = f"{y_axis_metric}_vs_{x_axis_param}_constrained.svg"
                
                # Save to a bytes buffer for download
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
                st.warning("No data available for the selected X and Y axes with the current constraints.")

    else:
        st.info("No numerical hyperparameters or metrics available for this analysis.")


    # --- Visualizations ---
    st.subheader("Metric Distributions and Relationships")

    # Reshape data for grouped analysis of output_X_metric
    output_metric_cols = [m for m in metrics if m.startswith('output_')]
    other_metrics = [m for m in metrics if not m.startswith('output_')]
    
    # Use all hyperparameters as ID variables for melting
    id_vars = hyperparameters + other_metrics
    
    df_melted = pd.DataFrame()
    if output_metric_cols:
        df_melted = df_cleaned.melt(
            id_vars=id_vars,
            value_vars=output_metric_cols,
            var_name='metric_full',
            value_name='metric_value'
        )
        # Extract neuron number and metric name from the full metric string
        extracted_data = df_melted['metric_full'].str.extract(r'output_(\d+)_(.*)')
        if extracted_data.shape[1] == 2:
            df_melted['output_neuron'] = pd.to_numeric(extracted_data[0])
            df_melted['metric_name'] = extracted_data[1]

    # Grouped plots for output_X_metric
    if not df_melted.empty and 'metric_name' in df_melted.columns:
        base_metrics = sorted(df_melted['metric_name'].unique())
        if base_metrics:
            st.write("### Grouped Analysis by Metric")
            selected_base_metrics = st.multiselect("Select Base Metrics for Grouped Analysis", options=base_metrics, default=base_metrics)

            for base_metric in selected_base_metrics:
                st.write(f"#### {base_metric}")
                df_subset = df_melted[df_melted['metric_name'] == base_metric]

                for cat_param in categorical_hyperparams:
                    fig = px.box(df_subset, x=cat_param, y='metric_value', color='output_neuron', title=f'{base_metric} vs. {cat_param}', points="all")
                    st.plotly_chart(fig, use_container_width=True)

                for num_param in numerical_hyperparams:
                    fig = px.scatter(df_subset, x=num_param, y='metric_value', color='output_neuron', title=f'{base_metric} vs. {num_param}')
                    st.plotly_chart(fig, use_container_width=True)

    # Plots for other metrics that don't fit the output_X_metric pattern
    if other_metrics:
        st.write("---")
        st.write("### Analysis for Other Metrics")
        for metric in other_metrics:
            st.write(f"#### Analysis for Metric: {metric}")
            for cat_param in categorical_hyperparams:
                fig = px.box(df_cleaned, x=cat_param, y=metric, title=f'{metric} vs. {cat_param}', points="all")
                st.plotly_chart(fig, use_container_width=True)
            for num_param in numerical_hyperparams:
                fig = px.scatter(df_cleaned, x=num_param, y=metric, title=f'{metric} vs. {num_param}', trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()
