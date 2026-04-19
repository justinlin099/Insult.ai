const CONFESSION_MAP = {
  "否認": 0,
  "坦承": 2,
  "未敘明": 0,
  "先否認後坦承": 1,
};

const ATTITUDE_MAP = {
  "良好": 3,
  "尚有悔意": 2,
  "尚可": 1,
  "未敘明": 0,
  "不佳": -1,
  "無悔意": -2,
};

const CONFESSION_OPTIONS = ["否認", "先否認後坦承", "坦承", "未敘明"];
const ATTITUDE_OPTIONS = ["無悔意", "不佳", "未敘明", "尚可", "尚有悔意", "良好"];

let webModel = null;

const form = document.getElementById("predict-form");
const resultEl = document.getElementById("result");
const statusEl = document.getElementById("status");
const predictBtn = document.getElementById("predict-btn");

function normalizeToken(token) {
  return token.trim();
}

function tokenizeInput(text) {
  const tokens = [];

  const quoted = text.matchAll(/[「\"]([^」\"]+)[」\"]/g);
  for (const match of quoted) {
    const value = normalizeToken(match[1]);
    if (value.length >= 2) tokens.push(value);
  }

  const splitTokens = text
    .replace(/[「」"'，,、。！？；：()（）\[\]{}]/g, " ")
    .split(/\s+/)
    .map(normalizeToken)
    .filter((token) => token.length >= 2);

  tokens.push(...splitTokens);

  const plain = normalizeToken(text);
  if (plain.length >= 2) {
    tokens.push(plain);
  }

  return tokens;
}

function getToggleValue(inputId, trueText, falseText) {
  const input = document.getElementById(inputId);
  return input.checked ? trueText : falseText;
}

function getSliderLabel(inputId, options) {
  const input = document.getElementById(inputId);
  const idx = Number(input.value);

  if (Number.isNaN(idx)) {
    return options[0];
  }

  const safeIndex = Math.max(0, Math.min(options.length - 1, idx));
  return options[safeIndex];
}

function setupToggle(inputId, stateId, trueText, falseText) {
  const input = document.getElementById(inputId);
  const state = document.getElementById(stateId);

  const update = () => {
    state.textContent = input.checked ? trueText : falseText;
  };

  input.addEventListener("change", update);
  update();
}

function setupSlider(inputId, valueId, options) {
  const input = document.getElementById(inputId);
  const label = document.getElementById(valueId);

  const update = () => {
    label.textContent = getSliderLabel(inputId, options);
  };

  input.addEventListener("input", update);
  update();
}

function initializeControls() {
  setupToggle("repeat_offender", "repeat_offender_state", "有", "無");
  setupToggle("mitigating_factor", "mitigating_factor_state", "有", "無");
  setupToggle("repeat_offender_while_on_parole", "repeat_offender_while_on_parole_state", "是", "否");
  setupToggle("repeat_offender_record", "repeat_offender_record_state", "有", "無");
  setupToggle("other_record", "other_record_state", "有", "無");

  setupSlider("confession", "confession_value", CONFESSION_OPTIONS);
  setupSlider("attitude", "attitude_value", ATTITUDE_OPTIONS);
}

function buildFeatureVector(vocabulary) {
  const vocabSize = Object.keys(vocabulary).length;
  const features = new Float64Array(8 + vocabSize);

  const text = (document.getElementById("insult_text").value || "").trim();
  const confession = getSliderLabel("confession", CONFESSION_OPTIONS);
  const attitude = getSliderLabel("attitude", ATTITUDE_OPTIONS);

  features[0] = getToggleValue("repeat_offender", "有", "無") === "有" ? 1 : 0;
  features[1] = getToggleValue("mitigating_factor", "有", "無") === "有" ? 1 : 0;
  features[2] = getToggleValue("repeat_offender_while_on_parole", "是", "否") === "是" ? 1 : 0;
  features[3] = getToggleValue("repeat_offender_record", "有", "無") === "有" ? 1 : 0;
  features[4] = getToggleValue("other_record", "有", "無") === "有" ? 1 : 0;
  features[5] = CONFESSION_MAP[confession] ?? 0;
  features[6] = ATTITUDE_MAP[attitude] ?? -2;
  features[7] = text.length;

  const tokens = tokenizeInput(text);
  for (const token of tokens) {
    const idx = vocabulary[token];
    if (idx !== undefined) {
      features[8 + idx] += 1;
    }
  }

  return { features, text };
}

function predictTree(features, tree) {
  let node = 0;

  while (tree.children_left[node] !== -1 && tree.children_right[node] !== -1) {
    const featureIdx = tree.feature[node];
    const threshold = tree.threshold[node];
    const value = features[featureIdx] ?? 0;

    node = value <= threshold ? tree.children_left[node] : tree.children_right[node];
  }

  return tree.value[node];
}

function predictForest(features, forest) {
  let sum = 0;

  for (const estimator of forest.estimators) {
    sum += predictTree(features, estimator);
  }

  return sum / forest.n_estimators;
}

function formatPrediction(predictionValue) {
  if (predictionValue > 10000) {
    if (predictionValue > 90000) {
      return `${Math.floor(predictionValue / 90000)} 月`;
    }

    return `${Math.floor(predictionValue / 1000)} 日`;
  }

  const rounded = Math.round(Math.floor(predictionValue) / 1000) * 1000;
  return `${rounded} 元`;
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function loadModel() {
  try {
    setStatus("模型載入中...");
    predictBtn.disabled = true;

    const response = await fetch("assets/model.json", { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    webModel = await response.json();
    setStatus("模型已就緒，可以開始預測");
    predictBtn.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus("模型載入失敗，請確認 assets/model.json 已存在");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();

  if (!webModel) {
    setStatus("模型尚未載入完成");
    return;
  }

  const { features, text } = buildFeatureVector(webModel.vectorizer.vocabulary);

  if (!text) {
    resultEl.textContent = "0 元";
    return;
  }

  const prediction = predictForest(features, webModel.forest);
  resultEl.textContent = formatPrediction(prediction);
});

initializeControls();
loadModel();
