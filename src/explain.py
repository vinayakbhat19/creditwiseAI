import numpy as np
import pandas as pd


def _clean_feature_name(name: str) -> str:
    name = name.replace("numeric__", "").replace("categorical__", "")
    return name.replace("_", " ")


def explain_prediction(pipeline, prepared_input, top_n=6):
    """
    Return top SHAP contributions for one applicant.
    Positive SHAP values push toward approval; negative values push toward rejection.
    """
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(prepared_input)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed_df)

    values = explanation.values[0]
    data_values = transformed_df.iloc[0].to_numpy()

    result = pd.DataFrame({
        "Feature": [_clean_feature_name(x) for x in feature_names],
        "SHAP Value": values,
        "Transformed Value": data_values,
    })

    result["Impact"] = np.where(
        result["SHAP Value"] >= 0,
        "Pushes toward approval",
        "Pushes toward rejection"
    )
    result["Absolute Impact"] = result["SHAP Value"].abs()

    return (
        result.sort_values("Absolute Impact", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
