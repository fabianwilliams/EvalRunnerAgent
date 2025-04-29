# 🧪 EvalRunnerAgent

A lightweight, Semantic Kernel-powered evaluation runner for testing LLM outputs against ground truth using similarity scoring.

---

## ✅ Current Features

- 🤖 **LLM Integration** — Uses OpenAI GPT-4o for natural language responses.
- 📊 **Semantic Scoring** — Cosine similarity on text embeddings to assess match.
- 🔁 **Retry Logic** — Auto-retries OpenAI calls up to 3 times.
- 💾 **Eval Logging** — Saves results with timestamp to `Data/eval_results_<timestamp>.json`
- 🧠 **Score Thresholding** — Pass/fail determined via configurable threshold.
- 🔐 **Configurable via `appsettings.json`**

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

