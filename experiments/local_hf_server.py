#!/usr/bin/env python
"""
Minimal OpenAI-compatible server for a local HuggingFace model, so llm_controller
(which already speaks the OpenAI provider) can drive a *local* model instead of
the Anthropic API or the remote Ollama tunnel. Heavy ML deps stay in this
process; the controller stays light and talks to it over HTTP.

Run in an env with torch + transformers (the project's `llm_stability` conda env):

    HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct PORT=8000 HF_DEVICE=cuda:0 \
        /home/andy/miniconda3/envs/llm_stability/bin/python experiments/local_hf_server.py

Point the controller at it (OpenAI provider):

    OPENAI_BASE_URL=http://localhost:8000/v1  OPENAI_API_KEY=local
    overlay: LLM_PROVIDER="openai", MODEL="<served HF id>"

Implements POST /v1/chat/completions with Qwen-native tool calling: a forced
single-tool request parses the model's <tool_call> JSON and returns it as an
OpenAI tool_calls response (with a JSON-object fallback). The weights are the
same cached HF checkpoints LLM_Stability/model_loader.py uses.
"""

import sys
# Some launch environments prepend the project's `llm_stability` conda env
# site-packages (a transformers install missing its `regex` dep) onto sys.path
# via PYTHONPATH. Drop it so transformers resolves to THIS interpreter's env.
sys.path = [p for p in sys.path if "/envs/llm_stability/" not in p]

import json
import os
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
PORT = int(os.environ.get("PORT", "8000"))
DEVICE = os.environ.get("HF_DEVICE", "cuda:0")

print(f"[local_hf] loading {MODEL_ID} on {DEVICE} ...", flush=True)
_tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if _tok.pad_token is None:
    _tok.pad_token = _tok.eos_token
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True,
).to(DEVICE).eval()
print(f"[local_hf] READY: {MODEL_ID}", flush=True)

_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _generate(messages, tools, max_tokens, temperature):
    kw = {"add_generation_prompt": True, "tokenize": False}
    if tools:
        kw["tools"] = tools
    prompt = _tok.apply_chat_template(messages, **kw)
    inputs = _tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-2),
            top_p=0.9,
            pad_token_id=_tok.pad_token_id or _tok.eos_token_id,
        )
    return _tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _extract_tool_args(text, required):
    """Pull the tool arguments out of the model's text (Qwen <tool_call> or raw JSON)."""
    for m in _TOOLCALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments", obj)
            if not required or all(k in args for k in required):
                return args
        except Exception:
            pass
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if not required or all(k in obj for k in required):
                return obj
        except Exception:
            pass
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        messages = req.get("messages", [])
        tools = req.get("tools")
        max_tokens = min(int(req.get("max_tokens") or 512), 1024)
        temperature = float(req.get("temperature", 0.7))

        text = _generate(messages, tools, max_tokens, temperature)

        if tools:
            fn = tools[0]["function"]
            required = fn.get("parameters", {}).get("required", [])
            args = _extract_tool_args(text, required)
            if args is not None:
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_" + uuid.uuid4().hex[:8],
                    "type": "function",
                    "function": {"name": fn["name"], "arguments": json.dumps(args)},
                }]}
                finish = "tool_calls"
            else:
                msg = {"role": "assistant", "content": text}  # controller will error clearly
                finish = "stop"
        else:
            msg = {"role": "assistant", "content": text}
            finish = "stop"

        self._send(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "created": 0,
            "model": req.get("model", MODEL_ID),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


if __name__ == "__main__":
    print(f"[local_hf] serving on http://127.0.0.1:{PORT}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
