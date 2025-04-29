# 🧪 EvalRunnerAgent

A lightweight, Semantic Kernel-powered evaluation runner for testing LLM outputs against ground truth using similarity scoring.

---

## ✅ Current Features

- 🤖 **LLM Integration** — Uses OpenAI GPT-4o or Local Ollama models for natural language responses.
- 📊 **Semantic Scoring** — Cosine similarity on text embeddings to assess match.
- 🔁 **Retry Logic** — Auto-retries LLM calls up to 3 times.
- 💾 **Eval Logging** — Saves results with timestamp to `Data/eval_results_<timestamp>.json`
- 🧠 **Score Thresholding** — Pass/fail determined via configurable threshold.
- 🔐 **Configurable via \`appsettings.json\`**

---

## 🔧 Example Output Summary

```bash
🚀 Starting Evaluation Agent...
🧠 Similarity Score: 0.9180
🧠 Similarity Score: 0.9231
🧠 Similarity Score: 0.8108

Results saved to Data/eval_results_20250429T151745.json
✅ Passed: 2 | ❌ Failed: 1
```

---

## ⚙️ Configuration

Edit `appsettings.json` to control model selection and thresholds:

```json
"Eval": {
  "UseLocalModels": true,          // true = use Ollama locally, false = use OpenAI
  "SimilarityThreshold": 0.75,
  "GroundTruthWeight": 0.7,
  "EvalCriteriaWeight": 0.3
},
"OpenAI": {
  "ApiKey": "sk-...",              // Your OpenAI key
  "ModelId": "gpt-4o",             // Any OpenAI-supported model
  "EmbeddingModel": "text-embedding-3-small"
},
"Ollama": {
  "Model": "llama3.3:70b-instruct",
  "Endpoint": "http://localhost:11434",
  "EmbeddingEndPoint": "http://localhost:11434",
  "EmbeddingModel": "nomic-embed-text:latest"
}
```

---

## 🚀 How to Run

### ✅ Local Mode (Ollama)

Ensure Ollama is running a compatible model (e.g., `llama3.3:70b-instruct`) locally:

```bash
ollama run llama3.3:70b-instruct
```

Then set in `appsettings.json`:

```json
"UseLocalModels": true
```

Run:

```bash
dotnet run
```

---

### 🌐 OpenAI Mode

Set in `appsettings.json`:

```json
"UseLocalModels": false
```

Ensure your OpenAI key and model ID are set correctly.

Run:

```bash
dotnet run
```

---

## 🎛 Optional: Override Similarity Threshold via CLI

To override the threshold (instead of editing the JSON):

```bash
dotnet run -- --threshold 0.78
```

This will apply the override just for that run.

---

## 📤 Output

Results are saved to a timestamped file in `Data`, e.g.:

```
Data/eval_results_20250429T214700.json
```

Each record includes:
- `Input`
- `GroundTruth`
- `ModelOutput`
- `SimilarityScore`
- `GroundTruthScore`
- `CriteriaScore`
- `Passed`: true/false
- `Notes`: ✅ Pass / ❌ Fail

---

## 🧠 Evaluation Logic

Final score is computed as:

```text
WeightedScore = (GroundTruthScore * GroundTruthWeight) + (CriteriaScore * EvalCriteriaWeight)
```

You control both weights and the threshold in config.

---

## 📸 Sample Output Snippet

```json
{
  "Input": "Calculate 5 multiplied by 6?",
  "ModelOutput": "5 × 6 = 30. So the answer is 30",
  "GroundTruth": "30",
  "SimilarityScore": 0.5886,
  "GroundTruthScore": 0.5835,
  "CriteriaScore": 0.6004,
  "ThresholdUsed": 0.75,
  "Passed": false,
  "Notes": "❌ Fail"
}
```

---

## 🛠️ TODO

- [ ] Add support for streaming model output
- [ ] Auto-load from multiple eval sets
- [ ] Support plugin-aware prompt evals (e.g., function-calling)
