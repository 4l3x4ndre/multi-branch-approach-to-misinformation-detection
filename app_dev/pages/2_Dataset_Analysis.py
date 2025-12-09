import streamlit as st
import pandas as pd
import json
from pathlib import Path
import seaborn as sns
import io
import matplotlib.pyplot as plt
import numpy as np
import sys
from corpus_truth_manipulation.config import CONFIG, PROCESSED_DATA_DIR
from src.utils import plot_utils

st.set_page_config(layout="wide")

st.markdown("# Dataset Analysis")

data_dir = PROCESSED_DATA_DIR
dataset_options = ["MMFakeBench_graphs", "cosmos_formatted", "xfacta_formatted", "MMFakeBench_graphs_stratified", 
                   "MMFakeBenchOriginal_graphs/test_original", "MMFakeBenchOriginal_graphs/val_original",
                   "cosmos_graphs"]

selected_dataset = st.selectbox("Select a dataset", dataset_options)


@st.cache_data
def load_data(dataset_name):
    data_path = data_dir / dataset_name
    all_data = []

    if "MMFakeBench" in dataset_name or "cosmos_graphs" in dataset_name:
        splits = ["train", "val", "test"]
        for split in splits:
            file_path = data_path / f"{split}_dbpedia_split.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        item["split"] = split
                    all_data.extend(data)
    elif dataset_name in ["cosmos_formatted", "xfacta_formatted"]:
        file_name = "cosmos.json" if dataset_name == "cosmos_formatted" else "xfacta.json"
        file_path = data_path / "test" / "source" / file_name
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                for item in data:
                    item['image_real'] = item['fake_cls'] in CONFIG.data.real_image_fake_cls or item['image_source'] in CONFIG.data.real_image_sources
                    item['claim_real'] = item['fake_cls'] in CONFIG.data.real_claim_fake_cls
                    item['mismatch'] = item['fake_cls'] in CONFIG.data.mismatch_fake_cls
                    item['overall_truth'] = item['fake_cls'] == 'original'
                all_data.extend(data)
    
    return pd.DataFrame(all_data)

if selected_dataset:
    df = load_data(selected_dataset)
    # Rename dataframe columns:
    df = df.rename(columns={
        "claim_real": "Claim Real",
        "image_real": "Image Real",
        "mismatch": "Mismatch",
        "overall_truth": "Overall Truth",
    })
    if not df.empty:
        st.write(f"Data loaded successfully. Columns: {df.columns.tolist()}")

        st.markdown("## Combined DataFrame")
        st.write(df)

        st.markdown("## Data Distribution")
        
        if selected_dataset == "cosmos_formatted":
            plot_cols = ["Mismatch"]
        else:
            plot_cols = ["Claim Real", "Image Real", "Mismatch", "Overall Truth"]
        
        # Filter out columns that are not in the dataframe
        plot_cols = [col for col in plot_cols if col in df.columns]
        
        if plot_cols:
            id_vars = ["split"] if "split" in df.columns else []
            df_melted = df.melt(id_vars=id_vars, value_vars=plot_cols, var_name="category", value_name="value")

            # Create a figure and axes for the plot
            figsize = (3.33, 10) if selected_dataset == "cosmos_formatted" else (10, 10)
            fig, ax = plot_utils.get_styled_figure_ax(figsize=figsize, aspect='none',grid=False)
            # fig, ax = plt.subplots()
            crosstab = pd.crosstab(df_melted['category'], df_melted['value'])
            crosstab.plot(kind='bar', stacked=True, ax=ax, color=plot_utils.DATASET_COLORS)
            plt.xticks(rotation=45)
            plot_utils.style_legend(ax, ncol=4, bbox_to_anchor=(0.5, 1.15))
            st.pyplot(fig)

            svg_buffer = io.BytesIO()
            fig.savefig(svg_buffer, format="pdf", bbox_inches="tight")
            svg_buffer.seek(0)

            st.download_button(
                label="Download plot as PDF",
                data=svg_buffer,
                file_name=f"data_distribution_{selected_dataset}.pdf",
                mime="application/pdf",
            )

            percentages = {}
            for category in crosstab.index:
                true_count = crosstab.loc[category].get(True, 0)
                false_count = crosstab.loc[category].get(False, 0)
                total = true_count + false_count
                if total > 0:
                    percentages[category] = (true_count / total) * 100
                else:
                    percentages[category] = 0
            
            md_string = ""                
            for category, percentage in percentages.items():
                md_string += f"- **{category}**: {percentage:.2f}% True\n"
            st.markdown(md_string)
            st.markdown(f"Number of totals samples: {len(df)}")

        if "split" in df.columns:
            st.markdown("## Data Distribution per Split")
            for split in df["split"].unique():
                st.markdown(f"### {split.capitalize()}")
                split_df_melted = df_melted[df_melted["split"] == split]
                
                if not split_df_melted.empty:
                    fig, ax = plot_utils.get_styled_figure_ax(aspect='none',grid=False)
                    crosstab = pd.crosstab(split_df_melted['category'], split_df_melted['value'])
                    crosstab.plot(kind='bar', stacked=True, ax=ax, color=plot_utils.DATASET_COLORS)
                    plot_utils.style_legend(ax, ncol=2, bbox_to_anchor=(0.5, 1.15))
                    st.pyplot(fig)

                    svg_buffer = io.BytesIO()
                    fig.savefig(svg_buffer, format="pdf", bbox_inches="tight")
                    svg_buffer.seek(0)

                    st.download_button(
                        label=f"Download {split} plot as PDF",
                        data=svg_buffer,
                        file_name=f"data_distribution_{selected_dataset}_{split}.pdf",
                        mime="application/pdf",
                        key=f"download_{split}",
                    )
    
                    percentages = {}
                    for category in crosstab.index:
                        true_count = crosstab.loc[category].get(True, 0)
                        false_count = crosstab.loc[category].get(False, 0)
                        total = true_count + false_count
                        if total > 0:
                            percentages[category] = (true_count / total) * 100
                        else:
                            percentages[category] = 0
                    
                    md_string = ""                    
                    for category, percentage in percentages.items():
                        md_string += f"- **{category}**: {percentage:.2f}% True\n"
                    st.markdown(md_string)                
                    st.markdown(f"Number of totals samples: {len(df[df['split'] == split])}")
                else:
                    st.info(f"No data for split: {split}")

        st.markdown("## Correlation Matrix (All Data)")

        # Create a figure and axes for the plot
        fig, ax = plot_utils.get_styled_figure_ax(grid=False)
        sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
        st.pyplot(fig)

        svg_buffer = io.BytesIO()
        fig.savefig(svg_buffer, format="pdf", bbox_inches="tight")
        svg_buffer.seek(0)

        st.download_button(
            label="Download correlation matrix as PDF",
            data=svg_buffer,
            file_name=f"correlation_matrix_{selected_dataset}.pdf",
            mime="application/pdf",
        )
        
        if "split" in df.columns:
            st.markdown("## Correlation Matrices per Split")

            for split in df["split"].unique():
                st.markdown(f"### {split.capitalize()}")
                split_df = df[df["split"] == split]

                # Create a figure and axes for the plot
                fig, ax = plot_utils.get_styled_figure_ax(grid=False)
                sns.heatmap(split_df.corr(numeric_only=True), annot=True, ax=ax)
                st.pyplot(fig)

                svg_buffer = io.BytesIO()
                fig.savefig(svg_buffer, format="pdf", bbox_inches="tight")
                svg_buffer.seek(0)

                st.download_button(
                    label=f"Download {split} correlation matrix as PDF",
                    data=svg_buffer,
                    file_name=f"correlation_matrix_{selected_dataset}_{split}.pdf",
                    mime="application/pdf",
                    key=f"download_corr_{split}",
                )
    else:
        st.warning(f"No data found for dataset '{selected_dataset}'. The directory might be empty or not contain the expected files.")
