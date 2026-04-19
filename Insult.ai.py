from flask import Flask, render_template, request
import joblib
import pandas as pd
import spacy


app = Flask(__name__)


# Load trained artifacts once on startup.
model = joblib.load("insult_fine_prediction_model.pkl")
vectorizer = joblib.load("insult_fine_prediction_vectorizer.pkl")

try:
    nlp = spacy.load("zh_core_web_sm")
except Exception:
    nlp = None


CONFESSION_MAP = {
    "否認": 0,
    "坦承": 2,
    "未敘明": 0,
    "先否認後坦承": 1,
}

ATTITUDE_MAP = {
    "良好": 3,
    "尚有悔意": 2,
    "尚可": 1,
    "未敘明": 0,
    "不佳": -1,
    "無悔意": -2,
}


columns = [
    "加重事由-累犯",
    "減輕事由",
    "行為人是否於緩刑中或假釋中再犯",
    "行為人有無「公然侮辱」之前案紀錄",
    "行為人有「公然侮辱」外之任何前案紀錄？",
    "是否坦承",
    "犯後態度",
    "侮辱字詞長度",
] + [f"vec_{i}" for i in range(len(vectorizer.get_feature_names_out()))]


def format_prediction(prediction_value):
    if prediction_value > 10000:
        if prediction_value > 90000:
            return f"{int(prediction_value / 90000)} 月"
        return f"{int(prediction_value / 1000)} 日"

    rounded = int(round(int(prediction_value) / 1000, 0) * 1000)
    return f"{rounded} 元"


def extract_insult_features(text):
    if nlp is None:
        return text

    doc = nlp(text)
    return " ".join([token.text for token in doc])


def predict_fine(form_data):
    text = form_data.get("insult_text", "").strip()
    if not text:
        return "0 元"

    repeat_offender = form_data.get("repeat_offender", "無")
    mitigating_factor = form_data.get("mitigating_factor", "無")
    repeat_offender_while_on_parole = form_data.get("repeat_offender_while_on_parole", "否")
    repeat_offender_record = form_data.get("repeat_offender_record", "無")
    other_record = form_data.get("other_record", "無")
    confession = form_data.get("confession", "否認")
    attitude = form_data.get("attitude", "良好")

    X_input = pd.DataFrame(
        {
            "加重事由-累犯": [1 if repeat_offender == "有" else 0],
            "減輕事由": [1 if mitigating_factor == "有" else 0],
            "行為人是否於緩刑中或假釋中再犯": [1 if repeat_offender_while_on_parole == "是" else 0],
            "行為人有無「公然侮辱」之前案紀錄": [1 if repeat_offender_record == "有" else 0],
            "行為人有「公然侮辱」外之任何前案紀錄？": [1 if other_record == "有" else 0],
            "是否坦承": [CONFESSION_MAP.get(confession, 0)],
            "犯後態度": [ATTITUDE_MAP.get(attitude, -2)],
            "侮辱字詞": [text],
        }
    )

    X_input["侮辱字詞特徵"] = X_input["侮辱字詞"].apply(extract_insult_features)
    X_input["侮辱字詞長度"] = X_input["侮辱字詞"].apply(len)
    X_input = X_input.drop(columns=["侮辱字詞"])

    X_text_vec = vectorizer.transform(X_input["侮辱字詞特徵"])
    X_text_vec_df = pd.DataFrame(
        X_text_vec.toarray(),
        columns=[f"vec_{i}" for i in range(X_text_vec.shape[1])],
    )

    X_input = X_input.drop(columns=["侮辱字詞特徵"])
    X_input = pd.concat([X_input.reset_index(drop=True), X_text_vec_df.reset_index(drop=True)], axis=1)

    X_full_input = pd.DataFrame(columns=columns)
    X_full_input = pd.concat([X_full_input, X_input], axis=0, ignore_index=True, join="outer")
    X_full_input = X_full_input.fillna(0).infer_objects()
    X_full_input.columns = X_full_input.columns.astype(str)

    prediction = model.predict(X_full_input)
    prediction_value = float(prediction[0])
    return format_prediction(prediction_value)


@app.route("/", methods=["GET", "POST"])
def index():
    defaults = {
        "insult_text": "",
        "repeat_offender": "無",
        "mitigating_factor": "無",
        "repeat_offender_while_on_parole": "否",
        "repeat_offender_record": "無",
        "other_record": "無",
        "confession": "否認",
        "attitude": "良好",
    }

    result = "0 元"
    form_values = defaults.copy()

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        result = predict_fine(form_values)

    return render_template("index.html", result=result, form=form_values)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)




