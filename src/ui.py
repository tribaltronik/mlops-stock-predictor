import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from test_predictor import compare_models

st.title("Stock Price Predictor - Model Comparison")

st.markdown("Compare multiple ML models for predicting JEDI.DE ETF prices")

if st.button("Run Model Comparison"):
    with st.spinner("Fetching data and training models..."):
        results = compare_models()

    if results:
        st.success("Model comparison completed!")

        # Create summary table
        summary_data = []
        for model, metrics in results.items():
            summary_data.append(
                {
                    "Model": model.replace("_", " ").title(),
                    "Accuracy (%)": f"{metrics['accuracy_percentage']:.2f}",
                    "R² Score": f"{metrics['r2_score']:.4f}",
                    "RMSE": f"{metrics['rmse']:.4f}",
                    "MAE": f"{metrics['mae']:.4f}",
                    "MAPE (%)": f"{metrics['mape']:.2f}",
                }
            )

        df = pd.DataFrame(summary_data)
        st.dataframe(df)

        # Bar chart for accuracy
        fig, ax = plt.subplots()
        models = [row["Model"] for row in summary_data]
        accuracies = [float(row["Accuracy (%)"]) for row in summary_data]
        ax.bar(models, accuracies)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Best model
        best_model = max(results.items(), key=lambda x: x[1]["accuracy_percentage"])
        st.markdown(
            f"**Best performing model:** {best_model[0].replace('_', ' ').title()} with {best_model[1]['accuracy_percentage']:.2f}% accuracy"
        )

    else:
        st.error("Unable to perform model comparison. Check data availability.")

st.markdown("---")
st.markdown(
    "This app compares Random Forest, Linear Regression, SVR, and Gradient Boosting models on JEDI.DE ETF data from 2020-2024."
)
