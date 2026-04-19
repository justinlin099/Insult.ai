import json
from pathlib import Path

import joblib


BASE_FEATURES = [
    "加重事由-累犯",
    "減輕事由",
    "行為人是否於緩刑中或假釋中再犯",
    "行為人有無「公然侮辱」之前案紀錄",
    "行為人有「公然侮辱」外之任何前案紀錄？",
    "是否坦承",
    "犯後態度",
    "侮辱字詞長度",
]


def export_tree(estimator):
    tree = estimator.tree_
    return {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, 0].tolist(),
    }


def main():
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(project_root / "insult_fine_prediction_model.pkl")
    vectorizer = joblib.load(project_root / "insult_fine_prediction_vectorizer.pkl")

    export_payload = {
        "version": 1,
        "base_features": BASE_FEATURES,
        "vectorizer": {
            "vocabulary": vectorizer.vocabulary_,
        },
        "forest": {
            "n_estimators": int(len(model.estimators_)),
            "estimators": [export_tree(est) for est in model.estimators_],
        },
    }

    output_path = output_dir / "model.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(export_payload, file, ensure_ascii=False, separators=(",", ":"))

    print(f"Exported web model to: {output_path}")


if __name__ == "__main__":
    main()
