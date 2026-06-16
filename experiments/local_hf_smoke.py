# Smoke-test overlay: drive the controller from the LOCAL HF model server
# (OpenAI provider -> http://localhost:8000/v1). Set when launching cli.py:
#   OPENAI_BASE_URL=http://localhost:8000/v1  OPENAI_API_KEY=local
# SMOKE ONLY -- K=2, not a full run.
LLM_PROVIDER   = "openai"
MODEL          = "Qwen/Qwen2.5-7B-Instruct"
WALLS          = []        # free-space S
MAX_ITERATIONS = 2
PLOT_IMAGE     = 0
RUN_NAME       = "local_hf_smoke"
