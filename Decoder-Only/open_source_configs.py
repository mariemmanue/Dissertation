"""
multi_prompt_configs_opensource.py
===================================
Open-source model version of multi_prompt_configs.py.
Optimized for HuggingFace local inference (Phi-4, Qwen, Llama, QwQ, etc.)

Key optimizations vs. original:
  1. Greedy decoding (do_sample=False) for non-reasoning models — removes noise,
     more consistent format compliance on annotation tasks.
  2. repetition_penalty=1.1 to prevent loops common in greedy decoding.
  3. pad_token_id set explicitly to suppress generation warnings.
  4. Default output_format is "json" — more reliable for smaller models.
  5. Lenient fallback parser (case-insensitive, dash-normalized) catches
     common formatting deviations from smaller models.
  6. Format reminder appended to the end of every user message.
  7. API backends (OpenAI, Gemini) removed — HuggingFace only.

Backends available:
  phi              microsoft/phi-4 (greedy)
  phi_reasoning    microsoft/Phi-4-reasoning (sampling, strips <think>)
  llama            meta-llama/* (greedy)
  qwen             Qwen/Qwen2.5-* (greedy)
  qwen3            Qwen/Qwen3-* no-thinking (greedy)
  qwen3_thinking   Qwen/Qwen3-* with thinking (sampling, strips <think>)
  qwq              Qwen/QwQ-32B (sampling, strips <think>)

Example commands:

  # Phi-4, zero-shot, JSON output (recommended for open-source)
  nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \\
    -n phi4_zs_json \\
    "python multi_prompt_configs_opensource.py \\
      --file Datasets/FullTest_Final.xlsx \\
      --model microsoft/phi-4 \\
      --backend phi \\
      --sheet PHI4_ZS_json \\
      --instruction_type zero_shot \\
      --extended \\
      --output_format json \\
      --output_dir Phi-4/data"

  # Qwen3 with thinking, few-shot CoT
  nlprun -g 1 -q sphinx -p standard -r 120G -c 4 \\
    -n qwen3_fscot \\
    "python multi_prompt_configs_opensource.py \\
      --file Datasets/FullTest_Final.xlsx \\
      --model Qwen/Qwen3-32B \\
      --backend qwen3_thinking \\
      --sheet QWEN3_FSCOT \\
      --instruction_type few_shot_cot \\
      --extended \\
      --dialect_legitimacy \\
      --output_format json \\
      --output_dir Qwen3/data"
"""

import os
import pandas as pd
import csv
import json
import torch
import re
import time
import math
import argparse
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================== GLOBAL TOKEN COUNTERS ====================

class TokenTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_call_count = 0

    def reset(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_call_count = 0


# ==================== BACKEND CLASSES ====================

@dataclass
class LLMBackend:
    """Abstract base. Each backend owns its own tokenizer."""
    name: str
    model: str

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        raise NotImplementedError

    def count_output_tokens(self, text: str) -> int:
        raise NotImplementedError

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        raise NotImplementedError


class PhiBackend(LLMBackend):
    """
    HuggingFace Phi-4 backend (local inference).

    Uses GREEDY decoding for annotation consistency.
    do_sample=False removes stochasticity that causes format deviations.
    repetition_penalty=1.1 prevents loops common in greedy on long outputs.

    CONFIDENCE SUPPORT: Global proxy only (avg max prob across tokens).
    """
    def __init__(self, model: str):
        super().__init__(name="phi", model=model)
        print(f"Loading Phi-4 from {model}...")
        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device
        self._last_confidence_data = None

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(formatted))
        except Exception:
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_tokens = 16000 if "cot" in instruction_type else 2000

        # GREEDY decoding: do_sample=False, no temperature/top_p
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text
            }

        # Strip markdown code fences
        if "```" in generated_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                generated_text = matches[-1].strip()
            else:
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        return generated_text


class PhiReasoningBackend(LLMBackend):
    """
    HuggingFace Phi-4-reasoning backend (local inference).

    Uses sampling with Phi-4-reasoning recommended params:
    temperature=0.8, top_p=0.95, top_k=50, max_new_tokens=32768.
    Strips <think>...</think> blocks from output before returning.

    CONFIDENCE SUPPORT: Global proxy only.
    """
    def __init__(self, model: str):
        super().__init__(name="phi_reasoning", model=model)
        print(f"Loading Phi-4-reasoning from {model}...")
        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device
        self._last_confidence_data = None

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(formatted))
        except Exception:
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Reasoning models need longer max for reasoning chain
        max_tokens = 32768 if "cot" in instruction_type else 16000

        # Keep sampling for reasoning chains — helps exploration
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text
            }

        # Strip reasoning blocks
        generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()

        # Strip markdown code fences
        if "```" in generated_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                generated_text = matches[-1].strip()
            else:
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        return generated_text


class LlamaBackend(LLMBackend):
    """
    HuggingFace Llama backend (local inference).

    Uses GREEDY decoding for annotation consistency.

    CONFIDENCE SUPPORT: Global proxy only.
    """
    def __init__(self, model: str):
        super().__init__(name="llama", model=model)
        print(f"Loading Llama from {model}...")
        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device
        self._last_confidence_data = None

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(formatted))
        except Exception:
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_tokens = 16000 if "cot" in instruction_type else 2000

        # GREEDY decoding
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text
            }

        if "```" in generated_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                generated_text = matches[-1].strip()
            else:
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        return generated_text


class QwenBackend(LLMBackend):
    """
    HuggingFace Qwen2.5 backend (local inference).

    Uses GREEDY decoding for annotation consistency.

    CONFIDENCE SUPPORT: Global proxy only.
    """
    def __init__(self, model: str):
        super().__init__(name="qwen", model=model)
        print(f"Loading Qwen from {model}...")
        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device
        self._last_confidence_data = None

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(formatted))
        except Exception:
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_tokens = 16000 if "cot" in instruction_type else 2000

        # GREEDY decoding
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text
            }

        if "```" in generated_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                generated_text = matches[-1].strip()
            else:
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        return generated_text


class Qwen3Backend(LLMBackend):
    """
    HuggingFace Qwen3 backend with thinking DISABLED (non-reasoning mode).

    Uses Qwen3 recommended non-thinking params: temperature=0.7, top_p=0.8, top_k=20.
    NOTE: Qwen3 non-thinking still benefits from slight sampling vs pure greedy.

    CONFIDENCE SUPPORT: Global proxy only.
    """
    def __init__(self, model: str):
        super().__init__(name="qwen3", model=model)
        print(f"Loading Qwen3 (non-thinking) from {model}...")
        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device
        self._last_confidence_data = None

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False
            )
            return len(self.tokenizer.encode(formatted))
        except Exception:
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_tokens = 16000 if "cot" in instruction_type else 2000

        # Qwen3 non-thinking recommended params (slight sampling helps instruction following)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text
            }

        if "```" in generated_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                generated_text = matches[-1].strip()
            else:
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        return generated_text


class Qwen3ThinkingBackend(LLMBackend):
    """
    HuggingFace Qwen3 backend with thinking ENABLED (reasoning mode).

    Uses Qwen3 recommended thinking params: temperature=0.6, top_p=0.95, top_k=20.
    Strips <think>...</think> blocks from output before returning.

    RECOMMENDED for annotation tasks — native thinking dramatically improves
    rule-following on fine-grained binary decisions.

    CONFIDENCE SUPPORT: Global proxy only.
    """
    def __init__(self, model: str):
        super().__init__(name="qwen3_thinking", model=model)
        print(f"Loading Qwen3 (thinking) from {model}...")
        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device
        self._last_confidence_data = None

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=True
            )
            return len(self.tokenizer.encode(formatted))
        except Exception:
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Long max for reasoning chain
        max_tokens = 32768 if "cot" in instruction_type else 16000

        # Qwen3 thinking recommended params
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text
            }

        # Strip reasoning blocks
        generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()

        if "```" in generated_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                generated_text = matches[-1].strip()
            else:
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        return generated_text


class QwQBackend(LLMBackend):
    """
    HuggingFace QwQ-32B backend (always-reasoning, based on Qwen2.5).

    QwQ always produces reasoning chains; strips <think>...</think> blocks.

    CONFIDENCE SUPPORT: Global proxy only.
    """
    def __init__(self, model: str):
        super().__init__(name="qwq", model=model)
        print(f"Loading QwQ from {model}...")
        self.model_id = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device
        self._last_confidence_data = None

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(formatted))
        except Exception:
            text = "\n".join(m["content"] for m in messages)
            return len(self.tokenizer.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_tokens = 32768 if "cot" in instruction_type else 16000

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text
            }

        # Strip reasoning blocks
        generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()

        if "```" in generated_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            matches = re.findall(pattern, generated_text, re.DOTALL)
            if matches:
                generated_text = matches[-1].strip()
            else:
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        return generated_text


# ==================== CONFIDENCE EXTRACTION ====================

def extract_feature_confidences(backend: LLMBackend, feature_names: list, parsed_labels: dict) -> dict:
    """
    Extract global proxy confidence from HuggingFace backends.
    Returns avg max probability across all generated tokens.
    Same score is assigned to ALL features (not feature-specific).
    """
    if not hasattr(backend, '_last_confidence_data') or not backend._last_confidence_data:
        return {}

    conf_data = backend._last_confidence_data
    if conf_data.get('type') != 'hf_scores':
        return {}

    scores = conf_data.get('scores')
    confidences = {}

    try:
        if not isinstance(scores, (list, tuple)):
            return {}

        max_probs = []
        for score_tensor in scores:
            if not hasattr(score_tensor, 'shape'):
                continue
            probs = torch.nn.functional.softmax(score_tensor[0], dim=-1)
            max_prob = torch.max(probs).item()
            max_probs.append(max_prob)

        avg_confidence = sum(max_probs) / len(max_probs) if max_probs else 0.5

        for feat in feature_names:
            confidences[feat] = avg_confidence

    except Exception as e:
        print(f"Warning: Could not extract HF scores: {e}")

    return confidences


# ==================== OUTPUT PARSING ====================

def extract_results(text: str, expected_features: list) -> tuple:
    """
    Parses Markdown-format model output.
    Expects:
        ### Analysis
        <free text reasoning>
        ### Results
        - feature-name: 1
        - feature-name: 0
    Returns (vals dict, rats dict, missing list).
    """
    vals = {}
    rats = {}

    analysis_match = re.search(r"### Analysis\s*(.*?)\s*### Results", text, re.DOTALL)
    global_rationale = analysis_match.group(1).strip() if analysis_match else "No analysis found."

    pattern = r"[-\*]\s*\*{0,2}([\w-]+)\*{0,2}\s*:\*{0,2}\s*(0|1)"
    for key, val in re.findall(pattern, text):
        if key in expected_features:
            vals[key] = int(val)
            rats[key] = f"See analysis: {global_rationale[:200]}..."

    missing = [f for f in expected_features if f not in vals]
    return vals, rats, missing


def extract_results_lenient(text: str, expected_features: list) -> tuple:
    """
    Lenient fallback parser for open-source model outputs.
    Case-insensitive, dash/underscore normalized.
    Catches common formatting deviations smaller models produce.
    """
    vals = {}
    rats = {}

    # More permissive: handles any list marker, any spacing, case-insensitive
    pattern = r"[-\*\+]?\s*\*{0,2}([\w\-_]+)\*{0,2}\s*:\s*\*{0,2}\s*(0|1)"
    for key_raw, val in re.findall(pattern, text, re.IGNORECASE):
        key_norm = key_raw.lower().strip()
        # Exact match first
        matched_feat = next(
            (f for f in expected_features if f.lower() == key_norm), None
        )
        # Dash/underscore normalized match
        if not matched_feat:
            matched_feat = next(
                (f for f in expected_features
                 if f.lower().replace("-", "").replace("_", "") ==
                    key_norm.replace("-", "").replace("_", "")),
                None
            )
        if matched_feat and matched_feat not in vals:
            vals[matched_feat] = int(val)
            rats[matched_feat] = ""

    missing = [f for f in expected_features if f not in vals]
    return vals, rats, missing


def extract_json_robust(text: str) -> dict | None:
    """
    Tries to parse a JSON object from model output.
    Falls back to regex extraction if json.loads fails
    (handles truncated JSON gracefully).
    """
    clean_text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"```", "", clean_text).strip()

    # Find the JSON object boundaries
    start = clean_text.rfind("{")
    end = clean_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_candidate = clean_text[start:end + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    data = {}
    # Simple flat labels: "key": 0 or "key": 1
    for key, val in re.findall(r'"([\w-]+)":\s*(0|1)', clean_text):
        data[key] = int(val)
    # Nested labels: "key": { "value": 0, ... }
    for key, val in re.findall(r'"([\w-]+)":\s*\{\s*"value":\s*(0|1)', clean_text):
        data[key] = int(val)

    return data if data else None


def parse_output(raw_str: str, features: list, output_format: str = "json") -> tuple:
    """
    Unified output parser with three-level fallback cascade:
      1. Requested format (JSON or Markdown)
      2. Other format (cross-format fallback)
      3. Lenient Markdown parser (catches small-model formatting deviations)

    Always returns (vals, rats, missing).
    """
    vals, rats, missing = {}, {}, list(features)

    if output_format == "json":
        data = extract_json_robust(raw_str)
        json_start = raw_str.find("{")
        pre_json_text = raw_str[:json_start].strip() if json_start > 0 else ""
        if data:
            for f in features:
                if f in data:
                    v = data[f]
                    vals[f] = v if isinstance(v, int) else int(v.get("value", 0))
                    if isinstance(v, dict) and v.get("rationale"):
                        rats[f] = str(v.get("rationale", ""))
                    elif pre_json_text:
                        rats[f] = f"See analysis: {pre_json_text[:200]}..."
                    else:
                        rats[f] = ""
            missing = [f for f in features if f not in vals]

        # Fallback 1: try Markdown
        if missing:
            md_vals, md_rats, _ = extract_results(raw_str, features)
            for f in list(missing):
                if f in md_vals:
                    vals[f] = md_vals[f]
                    rats[f] = md_rats.get(f, "")
                    missing.remove(f)

        # Fallback 2: lenient Markdown
        if missing:
            lv, lr, _ = extract_results_lenient(raw_str, features)
            for f in list(missing):
                if f in lv:
                    vals[f] = lv[f]
                    rats[f] = lr.get(f, "")
                    missing.remove(f)

    else:  # markdown
        vals, rats, missing = extract_results(raw_str, features)

        # Fallback 1: try JSON
        if missing:
            data = extract_json_robust(raw_str)
            if data:
                for f in list(missing):
                    if f in data:
                        v = data[f]
                        vals[f] = v if isinstance(v, int) else int(v.get("value", 0))
                        rats[f] = str(v.get("rationale", "")) if isinstance(v, dict) else ""
                        missing.remove(f)

        # Fallback 2: lenient Markdown
        if missing:
            lv, lr, _ = extract_results_lenient(raw_str, features)
            for f in list(missing):
                if f in lv:
                    vals[f] = lv[f]
                    rats[f] = lr.get(f, "")
                    missing.remove(f)

    return vals, rats, missing


# ==================== FEATURE LISTS ====================

MASIS_FEATURES = [
    "zero-poss", "zero-copula", "double-tense", "be-construction", "resultant-done",
    "finna", "come", "double-modal", "multiple-neg", "neg-inversion", "n-inv-neg-concord",
    "aint", "zero-3sg-pres-s", "is-was-gen", "zero-pl-s", "double-object", "wh-qu1", "wh-qu2",
]

EXTENDED_FEATURES = MASIS_FEATURES + [
    "existential-it", "demonstrative-them", "appositive-pleonastic-pronoun",
    "bin", "verb-stem", "past-tense-swap", "zero-rel-pronoun", "preterite-had", "bare-got",
]


# ==================== FEATURE BLOCKS ====================

MASIS_FEATURE_BLOCK = """
### LIST OF AAE MORPHOSYNTACTIC FEATURES ###

### Rule 1: zero-poss
**IF** a possessive relationship is expressed WITHOUT an overt SAE possessive morpheme ('s) or standard possessive pronoun form AND this possessive meaning is clearly licensed within the same clause, **THEN** label is 1.

Typical forms:
* Noun–noun juxtaposition (dad car, mama house)
* Bare or nonstandard possessive pronouns before a noun (they car, her brother kids)

+ Example: "That they dad boo."
  * Label is 1
  * Explanation: Nonstandard 'they' + 'dad boo' expresses possession

– Example: "That's their dad's car."
  * Label is 0
  * Explanation: All possessives fully marked in SAE

Note: If it is unclear whether two adjacent nouns form a possessive relationship or just a list/name (e.g., "school bus stop"), prefer 0 and mention ambiguity.

### Rule 2: zero-copula
**IF** a form of BE (is/are/was/were) that SAE requires is missing AND the utterance can be parsed as containing a clear subject–predicate relation (not just a list, heading, or obvious subordinate fragment), **THEN** label is 1.

This applies when BE is missing:
* Before a predicate (NP, AdjP, PP), OR
* Before a V-ing progressive, OR
* Before a preverbal future/near-future marker (finna, gonna)

+ Example: "She finna eat."
  * Label is 1
  * Explanation: Missing 'is' before preverbal marker 'finna'; SAE: "She is finna eat"

– Example: "No problem."
  * Label is 0
  * Explanation: Fragment with no recoverable subject–predicate BE slot

**COMMON FALSE POSITIVE GUARDS — mark 0 for these:**
* Participial or gerund phrases with no overt subject: "Getting that up." / "No father figure in his life, really good at a sport." — these are fragments, not zero-copula clauses.
* Clauses where the copula IS present but contracted or informal: "He's really tall" → 0.
* Prepositional/locative phrases that are complements of a prior clause: "Um, or in somebody yard doing something playing music" — the 'in somebody yard' is a PP complement, not a zero-copula predicate.
* Sentences where the subject is unclear or unrecoverable from the same utterance → prefer 0.

Note: If there is a recoverable subject and predicate slot that would host BE in SAE, mark 1 even if the utterance is short or informal. But do NOT create a subject–predicate reading where none exists.

### Rule 3: double-tense
**IF** a single lexical verb shows duplicated overt past-tense morphology (usually repeated -ed) within one word, **THEN** label is 1.

+ Example: "She likeded me the best."
  * Label is 1
  * Explanation: Duplicated -ed on 'likeded'

– Example: "She liked me the best."
  * Label is 0
  * Explanation: Single past-tense marker

Note: Spelling variants that do not clearly reflect two past morphemes should be 0 unless duplication is obvious.

### Rule 4: be-construction
**IF** uninflected 'be' appears as a finite verb expressing habitual, iterative, or generalized action/state, **THEN** label is 1.

Do not mark 'be' when it functions as an auxiliary of another tense or as an agreement error.

+ Example: "They be playing outside."
  * Label is 1
  * Explanation: Habitual 'be' expressing repeated activity

– Example: "They is playing outside."
  * Label is 0
  * Explanation: Agreement generalization, not habitual 'be'

Note: If 'be' could plausibly be part of a quoting frame or a fixed phrase without clear habitual reading, prefer 0 and explain.

### Rule 5: resultant-done
**IF** preverbal 'done' directly precedes a verb phrase and marks completed aspect for that event, **THEN** label is 1.

Adverbs or discourse markers may intervene. The following verb may be past-marked or bare, but the reading must be "already/completely finished."

+ Example: "He done already ate it."
  * Label is 1
  * Explanation: 'Done' marks completed eating

– Example: "They done it yesterday."
  * Label is 0
  * Explanation: Main verb 'done', not aspect marker

Note: If 'done' can be parsed as the main verb 'did' with no clear aspectual reading, prefer 0 and note ambiguity.

### Rule 6: finna
**IF** 'finna' (or 'finta', 'fitna', etc.) functions as a preverbal marker meaning 'about to / fixing to', creating an imminent or near-future reading, **THEN** label is 1.

+ Example: "We is finna eat."
  * Label is 1
  * Explanation: 'Finna' marks imminent future

### Rule 7: come
**IF** 'come' is used as a preverbal stance/evaluative marker introducing a verb phrase or V-ing form describing someone's behavior, attitude, or speech (not as a simple motion verb), **THEN** label is 1.

+ Example: "He come talking mess again."
  * Label is 1
  * Explanation: Stance 'come' + V-ing, not motion

– Example: "He come and talked."
  * Label is 0
  * Explanation: Motion verb + coordination

Note: If 'come' can be interpreted purely as motion toward a place with no clear stance/attitude meaning, prefer 0.

### Rule 8: double-modal
**IF** two modal-like elements appear in sequence in a single clause (e.g., 'might could', 'woulda could', 'used to could') and together scope over the following verb, **THEN** label is 1.

+ Example: "I might could go."
  * Label is 1
  * Explanation: Two modals in sequence

– Example: "I might go."
  * Label is 0
  * Explanation: Single modal

### Rule 9: multiple-neg
**IF** two or more negative elements (negative auxiliaries, adverbs, pronouns, or determiners) occur within the same clause or tightly integrated predicate and together express a single semantic negation (negative concord), **THEN** label is 1.

+ Example: "I ain't never heard of that."
  * Label is 1
  * Explanation: 'Ain't' + 'never' in same clause

– Example: "I ain't ever heard of that, not anyway."
  * Label is 0
  * Explanation: Single negation; "not anyway" is scalar/emphatic, not concord

Notes:
* This is the broad category covering both negative inversion (Rule 10) and negative concord (Rule 11).
* If either Rule 10 or Rule 11 applies, this feature (multiple-neg) must also be 1.

### Rule 10: neg-inversion
**IF** a negative auxiliary or marker occurs at the beginning of a sentence or clause and precedes the subject, with the subject immediately following it in that same clause, **THEN** label is 1.

+ Example: "Don't nobody like how they actin'."
  * Label is 1
  * Explanation: Negative auxiliary 'Don't' precedes subject 'nobody'

– Example: "Nobody don't like how they actin'."
  * Label is 0
  * Explanation: Subject precedes negative auxiliary; no inversion

– Example: "I ain't never doubted nobody."
  * Label is 0
  * Explanation: Negative elements not at beginning of clause

**DISAMBIGUATION from n-inv-neg-concord (Rule 11):**
These two features are mutually exclusive in any single clause.
* neg-inversion: NEGATIVE MARKER FIRST, then subject. Order = [NEG][SUBJECT][VERB]. Example: "Don't nobody..."
* n-inv-neg-concord: SUBJECT FIRST, then negative verb. Order = [NEG-SUBJECT][NEG-VERB]. Example: "Nobody don't..."
Check the ORDER: if the negative marker comes before the subject, it's neg-inversion. If the subject (which is itself negative) comes first, it's n-inv-neg-concord. NEVER mark both for the same clause.

**COMMON FALSE POSITIVE GUARD:** Multiple-negative sentences where the negative elements are in the middle of the clause (e.g., "I didn't do no report") are NOT neg-inversion. The inversion only applies when the negative auxiliary is clause-initial.

### Rule 11: n-inv-neg-concord
**IF** both the subject and the finite verb (or auxiliary) show overt negative marking AND the subject still comes first at the beginning of a sentence or clause AND together they express a single semantic negation, **THEN** label is 1.

+ Example: "Nobody don't wanna see that."
  * Label is 1
  * Explanation: Negative subject + negative auxiliary at clause start

– Example: "Nobody wanna see that."
  * Label is 0
  * Explanation: Subject negative, verb positive

– Example: "That's how nobody never seen."
  * Label is 0
  * Explanation: Negative elements not at beginning of clause

**DISAMBIGUATION from neg-inversion (Rule 10):**
* n-inv-neg-concord requires the SUBJECT to come FIRST and be NEGATIVE (nobody, nothing, ain't nobody, etc.), followed by a NEGATIVE VERB. Order = [NEG-SUBJECT][NEG-VERB].
* If the negative auxiliary precedes the subject, that's neg-inversion (Rule 10), not this feature.
* Both features require multiple-neg (Rule 9) to also be marked 1.

### Rule 12: aint
**IF** 'ain't' is used as a general negative auxiliary for BE, HAVE, or DO, or as a general clausal negator (rather than as a lexical verb), **THEN** label is 1.

+ Example: "She ain't here."
  * Label is 1
  * Explanation: Negated copula

– Example: "She isn't here."
  * Label is 0
  * Explanation: Not 'ain't'

### Rule 13: zero-3sg-pres-s
**IF** a 3rd person singular subject (he, she, it, this/that NP, nobody, somebody, etc.) co-occurs with a bare verb or uninflected auxiliary (do, have, walk, go) in PRESENT-TENSE meaning where SAE would require -s or does/has, **THEN** label is 1.

Exclude non-agreeing 'is/was' forms (those are Rule 14: is-was-gen).

+ Example: "She walk to they house."
  * Label is 1
  * Explanation: 3sg subject + bare 'walk' with present meaning; SAE: 'walks'

– Example: "They walk to their house."
  * Label is 0
  * Explanation: Plural subject; no 3sg requirement

Note: **If the bare verb occurs in reported speech, restarts, conditional/subjunctive context, or embedded clauses where the syntactic environment is ambiguous, prefer 0.** Only mark when the clause is clearly a finite present-tense declarative with 3sg subject.

**DISAMBIGUATION from verb-stem (Rule 22):**
* zero-3sg-pres-s: bare verb in a PRESENT-TENSE context with a 3sg subject. The time reference is NOW or habitual.
* verb-stem: bare verb in a PAST-TENSE context anchored by explicit time adverbs ("yesterday", "last week"), a prior past-tense verb in the same sequence, or an aspect marker like 'done'.
* ONE verb token cannot simultaneously be zero-3sg-pres-s AND verb-stem. Determine the tense reference FIRST, then assign the feature.
* If the tense is genuinely ambiguous (no explicit anchor), prefer zero-3sg-pres-s over verb-stem when there is a 3sg subject present.

### Rule 14: is-was-gen
**IF** 'is' or 'was' is used in a way that ignores SAE person/number agreement (e.g., with plural or 1st person subjects) in a finite clause, **THEN** label is 1.

Do NOT mark for existential 'it' constructions ("It was a fight," "It's people out here"), which are grammatical in SAE.

+ Example: "They was there."
  * Label is 1
  * Explanation: Plural subject + 'was'

– Example: "He was there."
  * Label is 0
  * Explanation: SAE-agreeing

Note: If 'was' may be part of a quoting frame or reported speech with unclear subject, prefer 0.

**DISAMBIGUATION from existential-it (Rule 18):**
When you see "it was [NP]..." or "it's [NP]...", ask: is 'it' a dummy existential subject introducing new entities (where SAE would say 'there was/were')? If yes → mark existential-it (Rule 18), do NOT also mark is-was-gen. The 'was' in "It was a couple of players that should've graduated" is part of the existential construction, not an agreement violation.
* is-was-gen requires a REAL plural or 1sg subject + 'is/was'.
* existential-it 'it' is a dummy subject with no referent — it is not a plural subject, so there is no agreement violation to mark.
* NEVER mark both existential-it and is-was-gen for the same 'it was/is' string.

### Rule 15: zero-pl-s
**IF** a noun that clearly has plural reference (from a quantifier, determiner, or context) surfaces without SAE plural -s AND the plural reading is local to the noun phrase, **THEN** label is 1.

+ Example: "She got them dog."
  * Label is 1
  * Explanation: Plural demonstrative 'them' + bare 'dog'

+ Example: "Two brother."
  * Label is 1
  * Explanation: Numeral 'two' licenses plural reading; bare 'brother' lacks -s; SAE: 'two brothers'

+ Example: "Sometime seven day."
  * Label is 1
  * Explanation: Numeral 'seven' + bare 'day'; SAE: 'seven days'

– Example: "A dogs."
  * Label is 0
  * Explanation: Article–noun mismatch, not AAE plural pattern

**STEP-BY-STEP DETECTION CHECK (use this sequence explicitly):**
1. Find every NOUN in the sentence.
2. For each noun, check: is there a numeral, quantifier, or plural demonstrative ('them', 'all', 'these', 'those') locally within the same NP that establishes plural reference?
3. Does that noun LACK the -s suffix that SAE would require?
4. If YES to both 2 and 3 → mark 1.
This feature has near-zero false positives but very high miss rate — actively search for it rather than waiting for it to be obvious.

Note: If plurality is only inferable from distant context and not clear in the NP itself, prefer 0.

### Rule 16: double-object
**IF** in a single clause, the subject and a following object pronoun (me, us, you, him, her, them) are coreferential (e.g., I…me, we…us, you…you) AND that pronoun is immediately followed by a noun phrase (NP) with no preposition (no to/for) AND together the verb + pronoun + NP express that the subject is obtaining, having, or wanting something for themself, **THEN** label is 1.

+ Example: "We had us a couple of beers."
  * Label is 1
  * Explanation: Subject 'we' = pronoun 'us'; self-benefactive

+ Example: "Soon as you get you some Scrabble tiles…"
  * Label is 1
  * Explanation: Subject 'you' = pronoun 'you'; self-benefit

– Example: "He gave me a book."
  * Label is 0
  * Explanation: Pronoun 'me' not coreferential with subject 'he'

– Example: "They got him a car."
  * Label is 0
  * Explanation: Subject 'they' ≠ pronoun 'him'

Notes:
* The crucial diagnostic is subject = object pronoun and a following NP with no preposition.
* Do not mark ordinary ditransitives where the pronoun refers to a different person.
* Exclude cases where the second element after the pronoun is not a full NP (e.g., tell you what, show you how).

### Rule 17a: wh-qu1 (WH-word + zero copula/DO deletion)
**IF** the string is a genuine WH-interrogative that makes a direct request for information AND SAE would require a form of BE or DO in that question, **THEN** label is 1.

This includes:
* Zero copula before a predicate or locative (Where she at?)
* Missing DO in WH-questions (What you want?, Where you go?)

+ Example: "Who you be talking to like that?"
  * Label is 1
  * Explanation: Missing 'are' between wh-word and subject; SAE: 'Who are you usually talking to like that?'

+ Example: "What you want?"
  * Label is 1
  * Explanation: Missing 'do'

– Example: "Where is she?"
  * Label is 0
  * Explanation: Auxiliary present

– Example: "I don't know what she wants."
  * Label is 0
  * Explanation: Not requesting information

Notes:
* Mark BOTH wh-qu1 AND zero-copula when the missing auxiliary is required for a WH-question in SAE.
* Do not mark for complements, fragments, or subordinate what/where clauses that are not clearly questions.

### Rule 17b: wh-qu2 (WH-word + non-standard inversion)
**IF** a WH-question or WH-clause departs from SAE subject–auxiliary inversion patterns (no inversion where SAE requires it OR inversion inside embedded WH-clauses where SAE keeps declarative order), **THEN** label is 1.

+ Example: "Where he is going?"
  * Label is 1
  * Explanation: Auxiliary follows subject in a main question

– Example: "I asked him if he could find her."
  * Label is 0
  * Explanation: Not a wh-question; standard word order

Note: Only mark wh-qu2 when WH-clause word order is non-standard relative to SAE.
"""

NEW_FEATURE_BLOCK = MASIS_FEATURE_BLOCK + """
### Rule 18: existential-it
**IF** 'it' functions as an existential/dummy subject in a construction where SAE would normally use 'there' to introduce an existential AND the following predicate introduces new entities, **THEN** label is 1.

+ Example: "It's people out here don't care."
  * Label is 1
  * Explanation: Existential 'it' + people; SAE: 'There are people'

– Example: "It is raining out here."
  * Label is 0
  * Explanation: Weather 'it' is grammatical in SAE

Note: If 'it' can be read as a true referential pronoun with a clear antecedent, prefer 0.

### Rule 19: demonstrative-them
**IF** 'them' is used directly before a noun as a demonstrative determiner meaning 'those' (not as an object pronoun), **THEN** label is 1.

+ Example: "Them shoes tight."
  * Label is 1
  * Explanation: 'Them' functions as demonstrative determiner

+ Example: "All them people."
  * Label is 1
  * Explanation: 'Them' as demonstrative even without immediately adjacent noun

– Example: "I like them."
  * Label is 0
  * Explanation: Object pronoun

+ Example: "See all them over there."
  * Label is 1
  * Explanation: 'Them' as demonstrative even when noun is ellipted

Note: 'Them' counts as demonstrative when it precedes a noun (even if a quantifier like 'all' intervenes) or when the noun is clearly recoverable from context (ellipted noun).

### Rule 20: appositive-pleonastic-pronoun
**IF** a subject or object NP is followed, **within the same clause**, by a clearly co-referential pronoun **in the same grammatical role** (subject + subject OR object + object), forming an appositive or pleonastic structure, **THEN** label is 1.

The pronoun must be **redundant**—the clause would be grammatical in SAE without it.

Fillers or pauses (e.g., 'uh', 'you know') may appear between NP and pronoun.

+ Example: "My dad, he told me it."
  * Label is 1
  * Explanation: NP 'my dad' + resumptive subject 'he' in same clause; clause would be grammatical as "My dad told me it"

+ Example: "The lawyer, I forgot about him."
  * Label is 0
  * Explanation: 'The lawyer' is left-dislocated topic; 'him' is the required object of "forgot about," not redundant

– Example: "A lot of people, you can tell they would tell me that."
  * Label is 0
  * Explanation: 'You' is subject of different clause; 'they' is also in different clause

**THE REDUNDANCY TEST — apply this before marking 1:**
Remove the pronoun. Read the result. Is it a complete, grammatical SAE sentence? If YES → the pronoun is pleonastic → mark 1. If NO (removing the pronoun breaks the sentence) → the pronoun is required → mark 0.
* "My dad, he told me it." → Remove 'he' → "My dad told me it." ✓ Grammatical → mark 1.
* "Cause I watch what my older brother all the trouble they got into." → 'they' is the subject of the embedded clause 'they got into' — it is required, not redundant → mark 0.

**CLAUSE BOUNDARY RULE — the NP and pronoun must be in the SAME clause:**
* If the NP is in one clause and the pronoun is in a DIFFERENT embedded/subordinate clause, do NOT mark appositive-pleonastic. The NP and pronoun must be co-present in the same finite clause.

**COMMON FALSE POSITIVE PATTERNS — mark 0 for these:**
* Long subject NP followed by pronoun that IS the grammatical subject: "My mother's side of the family we are close" — here 'my mother's side of the family' is the topic and 'we' is grammatical subject. Apply the redundancy test to disambiguate.
* "Neither one of 'em they do well" — 'they' is the grammatical subject that the rest of the clause requires; test whether removing 'they' leaves a grammatical clause.
* Sentences with disfluency/restarts where NP and pronoun are in separate discourse chunks rather than the same syntactic clause.

Note: The key diagnostic is whether removing the pronoun leaves a grammatical SAE clause. **Do not mark complex NPs with nested modifiers as appositive-pleonastic unless there is a separate, co-referential pronoun in the same grammatical role within the same clause.** If the pronoun is required as subject or object, it is not pleonastic.

### Rule 21: bin
**IF** BIN/been appears without an overt auxiliary 'have' AND expresses a long-standing or remote past state, **THEN** label is 1.

+ Example: "She been married."
  * Label is 1
  * Explanation: Long-standing state without 'have'

– Example: "She's been married for two years."
  * Label is 0
  * Explanation: Standard 'have been'

### Rule 22: verb-stem
**IF** a bare (uninflected) verb form serves as the finite verb of a clause that clearly refers to a past event based on local cues (explicit time adverbs, nearby past-tense anchors, or aspect markers like 'done') where SAE would require a past-tense form, **THEN** label is 1.

Local past-time cues include:
* Explicit time adverbs (yesterday, last week, ago)
* A past-tense verb in a coordinate or serial construction (e.g., "He ran and jump" – 'ran' anchors 'jump' as past)
* Aspect markers like 'done' that establish completed past

+ Example: "Yesterday he done walk to school."
  * Label is 1
  * Explanation: Bare 'walk' in past context; SAE: 'walked'

– Example: "He walk to school every day."
  * Label is 0
  * Explanation: Present habitual; possible zero-3sg-pres-s but not verb-stem

Note: **In coordinate or serial verb constructions (comma-separated verbs describing a sequence of events), if either verb is past-tense and another verb is bare, the bare verb counts as verb-stem because the first verb anchors the time reference as past.** If there is no explicit evidence that the event is past, prefer 0 and avoid assuming past meaning.

**DISAMBIGUATION from zero-3sg-pres-s (Rule 13):**
* verb-stem = bare form in a PAST context. Requires an explicit past anchor.
* zero-3sg-pres-s = bare form in a PRESENT context with a 3sg subject.
* If there is NO past-time anchor, do NOT label as verb-stem. Prefer zero-3sg-pres-s if there is a 3sg subject, or 0 if neither applies.

**DISAMBIGUATION from past-tense-swap (Rule 23):**
* verb-stem = NO overt tense morphology at all (the verb is a bare stem: walk, go, say).
* past-tense-swap = the verb HAS overt tense morphology, but it is the WRONG morphology for the position (e.g., 'seen' used as simple past, 'throwed' instead of 'threw').
* A verb cannot be simultaneously verb-stem AND past-tense-swap. Check: does the verb have any tense suffix or irregular past form? If yes → past-tense-swap. If truly bare with no morphology → verb-stem.

**COMMON FALSE POSITIVE GUARD:**
Bare verbs in subordinate clauses, conditional clauses ("if you go..."), and habitual present descriptions are NOT verb-stem even if the surrounding discourse is past. The bare verb's OWN clause must have past reference. Examples of false positives to avoid:
* "Whereas though after school you go there..." — 'go' is in a habitual/conditional frame, not past → 0.
* "Which means anything happen..." — 'happen' is in a present habitual embedded clause → 0.
* Present-tense narration with historic present is NOT verb-stem.

### Rule 23: past-tense-swap
**IF** an overtly non-SAE tense form is used as the main tense carrier of a clause AND the clause has clear simple-past or perfect/pluperfect reference, **THEN** label is 1.

Mark 1 when:
* A past participle is used as simple past (e.g., 'seen', 'done' for 'saw', 'did'), OR
* A regularized past is used where SAE requires irregular (e.g., 'throwed' for 'threw', 'droves' for 'drove', 'sung' for 'sang'), OR
* A past-tense form appears in any position where SAE requires a non-tensed form (bare infinitive after 'do/did/does', or distinct participle after 'have/had')

+ Example: "I seen him yesterday."
  * Label is 1
  * Explanation: Past participle 'seen' used as preterite; SAE: 'saw'

+ Example: "The dog had bit him before."
  * Label is 1
  * Explanation: Simple past 'bit' in pluperfect position; SAE: 'had bitten'

– Example: "I saw him yesterday."
  * Label is 0
  * Explanation: Standard preterite form

**MOST COMMONLY MISSED PATTERN — past participle as simple preterite:**
The single most under-detected case is a past participle (seen, done, come, gone, given, run) used as the only past-tense verb in a simple-past declarative clause, where SAE requires the simple past form (saw, did, came, went, gave, ran).
* "He just seen it happen again." → 'seen' as preterite for 'saw' → mark 1.
* "I done told you." → 'done' as preterite for 'did' → WAIT — check if 'done' is aspectual (resultant-done). If it precedes another verb, it may be resultant-done (Rule 5). If it is the sole main verb → past-tense-swap.
* Actively scan for 'seen', 'done', 'come', 'went' (past tense used where participle needed), 'brung', 'sung', 'throwed', 'droves', 'droved' as strong signals.

**DISAMBIGUATION from verb-stem (Rule 22):**
past-tense-swap requires the verb to have OVERT morphology — a suffix or irregular form. If the verb is bare with no morphology → verb-stem. The two features are mutually exclusive.

Note: This feature never applies to bare stems (those are verb-stem). If the verb has no overt tense morphology, mark 0 for past-tense-swap.

### Rule 24: zero-rel-pronoun
**IF** a finite clause modifies a noun and functions as a subject relative AND there is NO overt relative pronoun ('who', 'that', 'which') in subject position, **THEN** label is 1.

+ Example: "There are many mothers don't know their children."
  * Label is 1
  * Explanation: Clause modifying 'mothers' without 'who'; SAE: 'mothers who don't know'

+ Example: "It's a lot of stuff I learned from him."
  * Label is 1
  * Explanation: 'stuff [that] I learned' — wait, this is object gap, not subject relative → 0. But note: subject relatives are the target.

+ Example: "I knew people was doing that."
  * Label is 1 (if 'people' is modified by 'was doing that' with no overt 'who')
  * Explanation: Clause 'was doing that' modifies 'people' with no overt 'who'

– Example: "I think he left."
  * Label is 0
  * Explanation: That-deletion in complement clause, not subject relative

– Example: "The report you did yesterday" (object relative gap)
  * Label is 0
  * Explanation: 'you' is overt subject; the gap is in object position, not subject relative without pronoun

**STEP-BY-STEP DETECTION — this feature has near-zero detection rates; use this sequence:**
1. Find every noun or noun phrase in the sentence.
2. For each NP, ask: is there a finite clause (with its own subject + verb) immediately following that modifies that NP?
3. If yes, check: would SAE require 'who', 'that', or 'which' as the SUBJECT of that embedded clause?
4. Is 'who/that/which' ABSENT from the surface form?
5. If YES to 3 and 4 → mark 1.

**KEY DISTINCTION from that-deletion in complement clauses:**
* Subject relative: "people [who] don't know" — 'who' would be the SUBJECT of the embedded clause.
* Complement that-deletion: "I think [that] he left" — 'he' is overt; only the complementizer 'that' is missing, not a relative pronoun.
* Zero-rel-pronoun ONLY applies to the subject-relative case, not to complement clauses.

**This feature is VERY common in spontaneous AAE speech.** Sentences with long NPs followed by finite clauses are strong candidates. Actively look for it rather than waiting for obvious cases.

### Rule 25: preterite-had
**IF** 'had' plus a past verb is used to express a simple past event (with no clear 'past-before-past' meaning) AND there is no later past event anchoring a pluperfect reading, **THEN** label is 1.

Often appears with regularized/AAE-style past forms (had went, had ran, had did).

+ Example: "The alarm next door had went off a few minutes ago."
  * Label is 1
  * Explanation: Simple past meaning; no later reference event; 'went' is also a non-standard participle

– Example: "They had seen the movie before we arrived."
  * Label is 0
  * Explanation: True pluperfect (past-before-past); 'before we arrived' anchors pluperfect

**THE SUBSTITUTION TEST — apply before marking 1:**
Replace 'had [VERB]' with the simple past form of the verb. Does the sentence mean the same thing? If YES → preterite-had (mark 1). If changing to simple past loses the past-before-past meaning → NOT preterite-had (mark 0).
* "The alarm had went off." → "The alarm went off." Same meaning → mark 1.
* "They had cut my hours." → "They cut my hours." SAME meaning → BUT check for a second past event. If no second past event in context → mark 1. If the cutting clearly happened before another narrated event → mark 0.

**CRITICAL FALSE POSITIVE GUARD:**
Standard SAE pluperfect triggers this feature incorrectly. "And they had cut my hours to one or two" is standard SAE if it describes an event that preceded another narrated moment. Ask: is there a prior narrative event that this 'had' is anchoring against? If yes → SAE pluperfect → mark 0.

**SIGNAL OF TRUE PRETERITE-HAD:** The strongest signal is 'had' + non-standard past form ('had went', 'had did', 'had ran', 'had came', 'had brung'). Standard SAE uses the past PARTICIPLE after 'had' ('had gone', 'had done', 'had run'). If the form after 'had' is the simple past instead of the participle → very likely preterite-had → mark 1.

Note: If there is a clear second past event that makes 'had' plausibly pluperfect, prefer 0 and treat as SAE pluperfect.

### Rule 26: bare-got
**IF** 'got' functions as a present-tense possessive verb meaning 'have' AND there is NO overt 'have/has' or 'have got' construction in the same clause, **THEN** label is 1.

The subject has something right now (current possession), not a past 'got' event.

+ Example: "I got three kids."
  * Label is 1
  * Explanation: Present possession; SAE: 'I have three kids'

– Example: "I got a big paycheck last month."
  * Label is 0
  * Explanation: Simple past of 'get', not present possession

Note: Only mark bare-got when the clause clearly describes current possession and does not contain an overt 'have/has'. If 'got' can be read as simple past or as part of 'have got', prefer 0.
"""


# ==================== DISAMBIGUATION BLOCK ====================
# Injected at the END of the system message (recency advantage).
# Directly addresses the top confusion pairs identified in error analysis.

DISAMBIGUATION_BLOCK = """
### FEATURE DISAMBIGUATION REFERENCE ###
The following feature pairs are frequently confused. Before finalizing your labels,
check each pair that is relevant to the sentence.

---
**PAIR 1: verb-stem (Rule 22) vs. past-tense-swap (Rule 23) vs. zero-3sg-pres-s (Rule 13)**
These three features all involve non-standard verb morphology. Use this decision tree:

Step 1 — What is the TENSE CONTEXT of the clause?
  * PAST context (explicit time adverb, prior past verb, 'done' marker) → go to Step 2.
  * PRESENT context (habitual, current state, no past anchor) → check zero-3sg-pres-s (Rule 13).
  * Ambiguous → prefer zero-3sg-pres-s if 3sg subject present, else 0.

Step 2 — Does the verb have OVERT MORPHOLOGY?
  * Bare stem, no suffix, no irregular form (e.g., 'walk', 'say', 'go') → verb-stem (Rule 22).
  * Has overt but NON-STANDARD morphology (e.g., 'seen'/'done' as preterite, 'throwed', 'sung' for 'sang') → past-tense-swap (Rule 23).

A single verb cannot be verb-stem AND past-tense-swap simultaneously.
A single verb cannot be verb-stem AND zero-3sg-pres-s simultaneously.

---
**PAIR 2: neg-inversion (Rule 10) vs. n-inv-neg-concord (Rule 11)**
These are mutually exclusive. Check word ORDER:

  * [NEG-AUX] + [SUBJECT] + [VERB] = neg-inversion. Example: "Don't nobody know."
  * [NEG-SUBJECT] + [NEG-AUX/VERB] = n-inv-neg-concord. Example: "Nobody don't know."

Never mark both for the same clause. If multiple-neg (Rule 9) applies, decide which of Rule 10 or Rule 11 fits, or neither.

---
**PAIR 3: existential-it (Rule 18) vs. is-was-gen (Rule 14)**
When you see "it was/is [NP]..." ask: is 'it' a DUMMY existential subject (where SAE says 'there was/were')?
  * If yes → mark existential-it (Rule 18). Do NOT also mark is-was-gen. 'It' is not a plural subject.
  * If 'it' has a clear prior referent (a real pronoun with antecedent) AND 'was' disagrees with a real plural subject → is-was-gen (Rule 14).
  * These two features share the surface form "it was" but are mutually exclusive.

---
**PAIR 4: appositive-pleonastic-pronoun (Rule 20) vs. zero-rel-pronoun (Rule 24)**
These can co-occur but are often confused. Before marking appositive-pleonastic, run the REDUNDANCY TEST:
Remove the pronoun — is the result a complete grammatical SAE sentence?
  * YES → appositive-pleonastic (Rule 20).
  * NO (pronoun is required for the clause) → not appositive-pleonastic. Check zero-rel-pronoun separately.

For zero-rel-pronoun: look for NP + [finite clause without 'who/that/which' as subject].
These features look at different levels: appositive-pleonastic is about a redundant resumptive pronoun;
zero-rel-pronoun is about a missing relative pronoun that would be a SUBJECT.

---
**PAIR 5: preterite-had (Rule 25) vs. SAE pluperfect**
"had + past form" is the surface of BOTH. Use the SUBSTITUTION TEST:
  * Replace "had [VERB]" with simple past. Does meaning stay the same? → preterite-had (Rule 25).
  * STRONGEST SIGNAL: 'had' + non-standard past participle (had went, had did, had ran, had came) → preterite-had.
  * Standard SAE pluperfect uses correct participle form (had gone, had done, had run). If form is standard AND a pluperfect reading is available → NOT preterite-had → mark 0.

---
**ZERO-MARKING REMINDER (Rules 15, 24 — near-zero detection rates):**
These features are defined by the ABSENCE of something SAE requires. Models tend to miss them.
  * zero-pl-s (Rule 15): Actively scan EVERY noun for: numeral/quantifier + bare noun. Do not wait for it to be obvious.
  * zero-rel-pronoun (Rule 24): Actively scan EVERY NP for a following finite clause that modifies it without an overt 'who/that'. Do not wait for it to be obvious. It is VERY common in spontaneous speech.
"""

# ==================== ICL BLOCKS ====================

ICL_LABELS_ONLY_BLOCK = """
### FEW-SHOT TRAINING EXAMPLES ###
(for demonstration only; NOT the target utterance)

**Example 1:**
SENTENCE: "And my cousin family place turn into a whole cookout soon as it get warm, and when you step outside it's people dancing out on the sidewalk."

LABELS:
- zero-poss: 1
- zero-copula: 0
- double-tense: 0
- be-construction: 0
- resultant-done: 0
- finna: 0
- come: 0
- double-modal: 0
- multiple-neg: 0
- neg-inversion: 0
- n-inv-neg-concord: 0
- aint: 0
- zero-3sg-pres-s: 1
- is-was-gen: 0
- zero-pl-s: 0
- double-object: 0
- wh-qu1: 0
- wh-qu2: 0
- existential-it: 1
- demonstrative-them: 0
- appositive-pleonastic-pronoun: 0
- bin: 0
- verb-stem: 0
- past-tense-swap: 0
- zero-rel-pronoun: 0
- preterite-had: 0
- bare-got: 0

**Example 2:**
SENTENCE: "He throwed him a quick punch, then spin around and walk straight out the room like it was nothing."

LABELS:
- zero-poss: 0
- zero-copula: 0
- double-tense: 0
- be-construction: 0
- resultant-done: 0
- finna: 0
- come: 0
- double-modal: 0
- multiple-neg: 0
- neg-inversion: 0
- n-inv-neg-concord: 0
- aint: 0
- zero-3sg-pres-s: 0
- is-was-gen: 0
- zero-pl-s: 0
- double-object: 0
- wh-qu1: 0
- wh-qu2: 0
- existential-it: 0
- demonstrative-them: 0
- appositive-pleonastic-pronoun: 0
- bin: 0
- verb-stem: 1
- past-tense-swap: 1
- zero-rel-pronoun: 0
- preterite-had: 0
- bare-got: 0

**Example 3:**
SENTENCE: "He ain't never seen nothing move that fast before, then a week later he just seen it happen again right in front of him."

LABELS:
- zero-poss: 0
- zero-copula: 0
- double-tense: 0
- be-construction: 0
- resultant-done: 0
- finna: 0
- come: 0
- double-modal: 0
- multiple-neg: 1
- neg-inversion: 0
- n-inv-neg-concord: 0
- aint: 1
- zero-3sg-pres-s: 0
- is-was-gen: 0
- zero-pl-s: 0
- double-object: 0
- wh-qu1: 0
- wh-qu2: 0
- existential-it: 0
- demonstrative-them: 0
- appositive-pleonastic-pronoun: 0
- bin: 0
- verb-stem: 0
- past-tense-swap: 1
- zero-rel-pronoun: 0
- preterite-had: 0
- bare-got: 0

Use these as patterns for complete sentence annotation. Do NOT reuse these sentences when analyzing the new target utterance.
"""

ICL_COT_BLOCK = """
### FEW-SHOT TRAINING EXAMPLES ###
(for demonstration only; NOT the target utterance)

**Example 1:**
SENTENCE: "And my cousin family place turn into a whole cookout soon as it get warm, and when you step outside it's people dancing out on the sidewalk."

ANNOTATED LABELS (with rationales):
- zero-poss: 1
  RATIONALE: In 'my cousin family place', the nouns 'cousin' and 'family' express possession of 'place' without an overt 's morpheme. SAE would require 'my cousin's family place' or 'my cousin's family's place'. The possessive relationship is clear within the NP, so label is 1.

- zero-3sg-pres-s: 1
  RATIONALE: The subject 'my cousin family place' is a 3sg NP (singular compound). The present-tense verb 'turn' lacks the required -s. SAE: 'turns into'. Label is 1.

- existential-it: 1
  RATIONALE: In 'it's people dancing out on the sidewalk', 'it' functions as a dummy existential subject introducing new entities ('people'), where SAE would use 'there are people dancing'. Label is 1.

- multiple-neg: 0
  RATIONALE: No clause contains more than one negative element, so label is 0.

(all other features): 0

**Example 2:**
SENTENCE: "He throwed him a quick punch, then spin around and walk straight out the room like it was nothing."

ANNOTATED LABELS (with rationales):
- verb-stem: 1
  RATIONALE: The first verb 'throwed' anchors the time reference as past. The later bare verbs 'spin' and 'walk' occur in the same past-time sequence where SAE requires 'spun' and 'walked'. These are bare stems in past context, so label is 1.

- past-tense-swap: 1
  RATIONALE: 'Throwed' is a regularized past form used as simple past where SAE requires the irregular 'threw'. This is an overtly non-SAE tense form in a past-tense clause, so label is 1.

- double-object: 0
  RATIONALE: In 'throwed him a quick punch', the verb is followed by an indirect object 'him' and direct object 'a quick punch'. This is a standard SAE ditransitive frame. The pronoun 'him' is not coreferential with the subject 'he', so this is not the self-benefactive double-object construction. Label is 0.

(all other features): 0

**Example 3:**
SENTENCE: "He ain't never seen nothing move that fast before, then a week later he just seen it happen again right in front of him."

ANNOTATED LABELS (with rationales):
- multiple-neg: 1
  RATIONALE: In 'ain't never seen nothing', there are three negative elements in the same clause: 'ain't' (negative auxiliary), 'never' (negative adverb), and 'nothing' (negative pronoun). Together they express a single semantic negation (negative concord). Label is 1.

- aint: 1
  RATIONALE: 'Ain't' is used as a negative auxiliary for HAVE ('hasn't seen'), not as a main verb. This is the general negative auxiliary use, so label is 1.

- past-tense-swap: 1
  RATIONALE: In 'he just seen it happen again', 'seen' (a past participle form) is used as the only past-tense carrier where SAE requires the simple past 'saw'. This is an overtly non-SAE tense form in simple-past position, so label is 1.

- verb-stem: 0
  RATIONALE: No bare stem verb serves as the finite past predicate. 'Seen' is a past participle used as simple past (captured by past-tense-swap), and 'happen' is non-finite in 'seen it happen'. Label is 0.

(all other features): 0

Use these as patterns: for each feature, identify the exact grammatical evidence and apply the decision rule. Do NOT reuse these sentences when analyzing the new target utterance.
"""


# ==================== UTILITIES ====================

def drop_features_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["FEATURES", "Source"], errors="ignore")


def strip_examples_from_block(block: str) -> str:
    """
    Remove annotated example lines from a feature block (zero-shot conditions).
    Handles ASCII hyphen, en-dash (U+2013), and em-dash (U+2014).
    """
    example_line_re = re.compile(
        r'^[+\-\u2013\u2014]\s+(?:Example|Miss)', re.UNICODE
    )
    sub_bullet_re = re.compile(
        r'^\*\s+(?:Label is|Explanation:)', re.UNICODE
    )
    stripped_lines = []
    for line in block.splitlines():
        s = line.lstrip()
        if example_line_re.match(s) or sub_bullet_re.match(s):
            continue
        stripped_lines.append(line)
    return "\n".join(stripped_lines)


def has_usable_context(left_context, right_context) -> bool:
    return bool(_clean_ctx(left_context) or _clean_ctx(right_context))


def _clean_ctx(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    return s if s else None


def format_context_block(left_context, right_context) -> str:
    left_context = _clean_ctx(left_context)
    right_context = _clean_ctx(right_context)
    lines = []
    if left_context:
        sents = [s for s in left_context.split("\n") if s.strip()]
        for i, s in enumerate(sents):
            label = i - len(sents)
            lines.append(f"PREV [{label}]: {s}")
    if right_context:
        sents = [s for s in right_context.split("\n") if s.strip()]
        for i, s in enumerate(sents, 1):
            lines.append(f"NEXT [+{i}]: {s}")
    return "\n".join(lines)


# ==================== PROMPT BUILDERS ====================

def build_system_msg(
    instruction_type: str,
    dialect_legitimacy: bool,
    use_context: bool,
    base_feature_block: str,
    include_examples_in_block: bool,
) -> dict:
    """
    Build the full system message.
    Static content: framing, procedure, constraints, feature rules, ICL examples.
    """

    if dialect_legitimacy:
        intro = (
            "You are a highly experienced sociolinguist and expert annotator of African American English (AAE).\n"
            "AAE is a rule-governed, systematic language variety. You must analyze the input according to AAE's "
            "internal grammatical rules, not Standard American English (SAE) norms.\n"
            "Treat African American English and Standard American English as equally valid.\n"
            "Do not 'correct' or rate sentences. Your only task is to decide, for each feature, whether its "
            "definition matches this utterance (1) or not (0).\n"
            "Informal or slangy English that could be spoken by many speakers (regardless of race) does NOT "
            "automatically count as AAE. Only mark a feature as 1 if the utterance clearly matches the specific "
            "AAE morphosyntactic pattern described in the decision rule.\n\n"
        )
    else:
        intro = (
            "You are a linguist analyzing morphosyntactic features in a variety of English often referred to as "
            "African American English (AAE).\n"
            "Your goal is to identify specific grammatical constructions in the input utterance, comparing them "
            "to Standard American English (SAE) where relevant.\n\n"
        )

    procedure = (
        "Your task is to make strictly binary decisions (1 = present, 0 = absent) about a fixed list of AAE "
        "morphosyntactic features for a single target utterance.\n\n"
        "### PROCEDURE ###\n"
        "Follow these steps:\n\n"
        "**1. CLAUSE & TENSE ANALYSIS (global):**\n"
        "* Identify the main clause of the TARGET UTTERANCE\n"
        "* Identify its grammatical subject(s), finite verb(s), and any auxiliaries\n"
        "* Identify tense/aspect markers (e.g., done, BIN, had) and overt temporal expressions\n"
        "* Identify the scope and pattern of negation and any embedded clauses\n\n"
        "**2. FEATURE-BY-FEATURE EVALUATION:**\n"
        "* For each feature, check whether the TARGET UTTERANCE satisfies ALL parts of that rule\n"
        "* For tense-related features, ask:\n"
        "  - What tense/aspect form does the verb show? (bare, simple past, past participle, etc.)\n"
        "  - What form does SAE require in this syntactic position?\n"
        "  - Does the mismatch fit the AAE pattern described in the rule?\n\n"
        "**3. MULTIPLE INTERPRETATIONS:**\n"
        "* If more than one grammatical analysis is genuinely possible, briefly acknowledge the strongest alternative\n"
        "* **Prefer precision over recall**: if the utterance does not provide enough grammatical evidence to "
        "confidently apply the rule, output 0\n"
        "* Do NOT treat disfluencies or casual phrasing as ambiguity unless they obscure the relevant syntactic environment\n\n"
    )

    constraints = (
        "### EXPLICIT EVALUATION CONSTRAINTS ###\n"
        "* Analyze ONLY syntax, morphology, and clause structure\n"
        "* Base each decision only on the feature definitions and the words in the target utterance\n"
        "* Do not infer extra events or repair the sentence\n"
        "* Informal, conversational, or slang does NOT by itself imply an AAE feature\n\n"
        "**SUBJECT DROPS IN SPONTANEOUS SPEECH:**\n"
        "* If a clause has a clearly recoverable subject from the SAME utterance or immediately preceding clause, "
        "you may treat that subject as syntactically present\n"
        "* If no specific subject is clearly recoverable, prefer 0\n\n"
        "**FOR TENSE-RELATED FEATURES:**\n"
        "* First determine intended reference time using explicit time adverbs, aspect markers, and local context\n"
        "* Then compare the verb form to what SAE would require in that context\n"
        "* Only mark 1 when the mismatch fits the defined AAE pattern\n\n"
    )

    ctx_instructions = ""
    if use_context:
        ctx_instructions = (
            "**CONTEXT USE (if provided in the user message):**\n"
            "* You may use PREV/NEXT sentence context ONLY to resolve:\n"
            "  - Recoverable subject in the SAME utterance chain\n"
            "  - Explicit temporal reference (past vs present) when stated in context\n"
            "* Do NOT use context to invent missing words or infer events not stated\n\n"
        )

    if "cot" in instruction_type:
        cot_instruction = (
            "### CHAIN-OF-THOUGHT REQUIREMENT ###\n"
            "Provide DETAILED step-by-step reasoning for each feature:\n"
            "1. Quote the exact substring from the sentence that is relevant\n"
            "2. Explain the grammatical pattern you observe\n"
            "3. State what SAE would require in this position\n"
            "4. Explain why the AAE rule applies or doesn't apply\n\n"
            "Your analysis should be thorough (5-10 sentences per feature that is present).\n\n"
        )
    else:
        cot_instruction = (
            "### REASONING REQUIREMENT ###\n"
            "Provide BRIEF reasoning for each decision (1-2 sentences per feature).\n"
            "Focus on the key grammatical evidence.\n\n"
        )

    effective_include_examples = include_examples_in_block
    if instruction_type in ["zero_shot", "zero_shot_cot"]:
        effective_include_examples = False
    feature_rules = base_feature_block if effective_include_examples else strip_examples_from_block(base_feature_block)

    icl_block = ""
    if instruction_type == "few_shot":
        icl_block = ICL_LABELS_ONLY_BLOCK
    elif instruction_type == "few_shot_cot":
        icl_block = ICL_COT_BLOCK

    # DISAMBIGUATION_BLOCK placed AFTER ICL (last = most salient at inference time)
    content = "".join([
        intro,
        procedure,
        constraints,
        ctx_instructions,
        cot_instruction,
        feature_rules,
        icl_block,
        DISAMBIGUATION_BLOCK,
    ])

    return {"role": "system", "content": content}


def build_user_msg(
    utterance: str,
    features: list,
    output_format: str,
    instruction_type: str = "zero_shot",
    context_block: str | None = None,
) -> str:
    """
    Build the per-sentence user message.
    Includes a format reminder at the end — helps smaller models stay on track.
    """
    feature_list_str = ", ".join(f'"{f}"' for f in features)

    context_section = ""
    if context_block:
        context_section = (
            "### CONTEXT ###\n"
            "(Use ONLY to resolve subject reference and tense, NOT to infer events)\n\n"
            f"{context_block}\n\n"
        )

    if output_format == "markdown":
        output_instructions = (
            "### OUTPUT INSTRUCTIONS ###\n"
            "Structure your response in two parts:\n\n"
            "**PART 1 — `### Analysis`**\n"
            "Walk through the sentence systematically. For features that are present (1), quote the exact "
            "substring and explain the grammatical pattern. For features that are absent (0), briefly state "
            "why the rule is not met.\n\n"
            "**PART 2 — `### Results`**\n"
            "After your analysis, output ALL features as a bulleted list using EXACTLY this format "
            "(no bold, no extra text on the label line):\n"
            "- feature-name: 1\n"
            "- feature-name: 0\n\n"
            "Do NOT bold the feature names. Do NOT add explanations on the same line as the label. "
            "Every feature in the list below must appear exactly once.\n"
        )
    else:
        if "cot" in instruction_type:
            output_instructions = (
                "### OUTPUT INSTRUCTIONS ###\n"
                "Structure your response in two parts:\n\n"
                "**PART 1 — Analysis**\n"
                "Provide detailed step-by-step analysis. For each feature, quote the relevant substring, "
                "explain the grammatical pattern, state what SAE would require, and explain why the AAE rule "
                "applies or doesn't.\n\n"
                "**PART 2 — JSON Labels**\n"
                "After your analysis, output ALL final labels as a single flat JSON object with integer values:\n"
                '  {"feature-name": 0, "feature-name": 1, ...}\n\n'
                "Do NOT include rationales or nested objects inside the JSON. "
                "Every feature in the list below must appear exactly once in the JSON.\n"
            )
        else:
            output_instructions = (
                "### OUTPUT INSTRUCTIONS ###\n"
                "Structure your response in two parts:\n\n"
                "**PART 1 — Analysis**\n"
                "Briefly walk through the sentence. For features that are present (1), note the key evidence. "
                "For features that are absent (0), briefly state why.\n\n"
                "**PART 2 — JSON Labels**\n"
                "After your analysis, output ALL final labels as a single flat JSON object with integer values:\n"
                '  {"feature-name": 0, "feature-name": 1, ...}\n\n'
                "Do NOT include rationales or nested objects inside the JSON. "
                "Every feature in the list below must appear exactly once in the JSON.\n"
            )

    # Format reminder at end — recency bias helps smaller models stay on format
    format_reminder = (
        f"\nREMINDER: Your JSON (or Results list) must include ALL {len(features)} features exactly once "
        f"with integer values 0 or 1 only. Features: [{feature_list_str}]"
    )

    return (
        f"{context_section}"
        f"### TARGET UTTERANCE ###\n"
        f'"{utterance}"\n\n'
        f"### FEATURES TO LABEL ###\n"
        f"[{feature_list_str}]\n\n"
        f"{output_instructions}"
        f"{format_reminder}"
    )


def build_messages(
    utterance: str,
    features: list,
    base_feature_block: str,
    instruction_type: str,
    include_examples_in_block: bool,
    output_format: str = "json",
    use_context: bool = False,
    left_context: str | None = None,
    right_context: str | None = None,
    context_mode: str = "single_turn",
    dialect_legitimacy: bool = False,
) -> tuple[list[dict], str]:
    """
    Assembles the full messages list for a single sentence.
    Returns (messages, arm_used).
    """
    system_msg = build_system_msg(
        instruction_type=instruction_type,
        dialect_legitimacy=dialect_legitimacy,
        use_context=use_context,
        base_feature_block=base_feature_block,
        include_examples_in_block=include_examples_in_block,
    )

    context_block = None
    if use_context:
        cb = format_context_block(left_context, right_context)
        if cb.strip():
            context_block = cb

    user_content = build_user_msg(
        utterance=utterance,
        features=features,
        output_format=output_format,
        instruction_type=instruction_type,
        context_block=context_block,
    )
    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg], context_mode


# ==================== USAGE SUMMARY ====================

def print_final_usage_summary(tracker: TokenTracker):
    total_tokens = tracker.total_input_tokens + tracker.total_output_tokens
    print(f"Total Inference Calls: {tracker.api_call_count}")
    print(f"Input Tokens:          {tracker.total_input_tokens}")
    print(f"Output Tokens:         {tracker.total_output_tokens}")
    print(f"Total Tokens:          {total_tokens}")


# ==================== MODEL QUERY ====================

def query_model(
    backend: LLMBackend,
    tracker: TokenTracker,
    sentence: str,
    features: list,
    base_feature_block: str,
    instruction_type: str,
    include_examples_in_block: bool,
    output_format: str = "json",
    use_context: bool = False,
    left_context: str | None = None,
    right_context: str | None = None,
    context_mode: str = "single_turn",
    dialect_legitimacy: bool = False,
    dump_prompt: bool = False,
    dump_prompt_path: str | None = None,
    dump_counter: int = 0,
    dump_first_n: int = 1,
    sentence_idx: int = 0,
    max_retries: int = 3,
    base_delay: int = 5,
) -> tuple[str | None, str]:

    messages, arm_used = build_messages(
        utterance=sentence,
        features=features,
        base_feature_block=base_feature_block,
        instruction_type=instruction_type,
        include_examples_in_block=include_examples_in_block,
        output_format=output_format,
        use_context=use_context,
        left_context=left_context,
        right_context=right_context,
        context_mode=context_mode,
        dialect_legitimacy=dialect_legitimacy,
    )

    # Prompt dump
    should_dump = dump_prompt and (dump_first_n == 0 or dump_counter < dump_first_n)
    if should_dump:
        print("\n" + "=" * 80)
        print(f"PROMPT DUMP #{dump_counter + 1} (Sentence idx: {sentence_idx})")
        print("=" * 80)
        print(f"\n   TARGET SENTENCE:  \"{sentence[:100]}{'...' if len(sentence) > 100 else ''}\"")
        print(f"\n   use_context: {use_context} | context_mode: {context_mode} | arm: {arm_used}")
        if left_context and str(left_context).strip():
            print(f"   LEFT:  \"{str(left_context).strip()[:100]}\"")
        if right_context and str(right_context).strip():
            print(f"   RIGHT: \"{str(right_context).strip()[:100]}\"")
        model_name = getattr(backend, 'model_id', backend.model)
        if not isinstance(model_name, str):
            model_name = backend.name
        payload = {
            "backend": backend.name,
            "model": model_name,
            "sentence_idx": sentence_idx,
            "messages": messages,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("=" * 80 + "\n")
        if dump_prompt_path:
            base, ext = os.path.splitext(dump_prompt_path)
            output_path = f"{base}_{dump_counter + 1}{ext}" if dump_first_n != 1 else dump_prompt_path
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Prompt saved to: {output_path}\n")

    input_tokens = backend.count_tokens(messages)

    for attempt in range(max_retries):
        try:
            output_text = backend.call(messages, instruction_type=instruction_type)
            output_tokens = backend.count_output_tokens(output_text)

            tracker.total_input_tokens += input_tokens
            tracker.total_output_tokens += output_tokens
            tracker.api_call_count += 1

            print(
                f"Call #{tracker.api_call_count} | "
                f"Input: {input_tokens} | Output: {output_tokens} | "
                f"Running total: {tracker.total_input_tokens + tracker.total_output_tokens}"
            )

            return output_text, arm_used

        except Exception as e:
            msg = str(e).lower()
            is_last = (attempt == max_retries - 1)

            if "cuda out of memory" in msg or "oom" in msg:
                print(f"OOM error on sentence {sentence_idx}. Cannot retry. Skipping.")
                return None, arm_used

            wait_time = base_delay * (2 ** attempt)
            print(f"Error (attempt {attempt + 1}/{max_retries}): {str(e)[:150]}")
            if not is_last:
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None, arm_used

    return None, arm_used


# ==================== MAIN ====================

def main():
    print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"DEBUG: torch.version.cuda        = {torch.version.cuda}")

    parser = argparse.ArgumentParser(
        description="AAE feature annotation — open-source HuggingFace models only."
    )
    parser.add_argument("--file",             type=str, required=True,  help="Input Excel file path")
    parser.add_argument("--gold",             type=str, default=None,   help="Gold labels CSV/Excel file")
    parser.add_argument("--sheet",            type=str, required=True,  help="Sheet name for predictions output")
    parser.add_argument("--extended",         action="store_true",      help="Use extended 27-feature set")
    parser.add_argument("--context",          action="store_true",      help="Include prev/next sentence context")
    parser.add_argument("--instruction_type", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"])
    parser.add_argument("--block_examples",   action="store_true",      help="Keep examples in feature block")
    parser.add_argument("--dialect_legitimacy", action="store_true",    help="Frame AAE as rule-governed")
    parser.add_argument("--output_dir",       type=str, required=True,  help="Output directory")
    parser.add_argument("--dump_prompt",      action="store_true",      help="Print prompt to stdout")
    parser.add_argument("--dump_prompt_path", type=str, default=None,   help="Write prompt JSON to path")
    parser.add_argument("--dump_first_n",     type=int, default=1,      help="Dump first N prompts (0=all)")
    parser.add_argument("--context_mode",     type=str, default="single_turn",
                        choices=["single_turn", "wide"])
    parser.add_argument("--backend",          type=str, default="phi",
                        choices=["phi", "phi_reasoning", "llama",
                                 "qwen", "qwen3", "qwen3_thinking", "qwq"])
    parser.add_argument("--model",            type=str, default="microsoft/phi-4",
                        help="HuggingFace model ID (e.g. microsoft/phi-4, Qwen/Qwen3-32B)")
    # Default output_format is json — more reliable for open-source models
    parser.add_argument("--output_format",    type=str, default="json",
                        choices=["json", "markdown"],
                        help="Output format: 'json' (recommended for open-source) or 'markdown'")

    args = parser.parse_args()
    tracker = TokenTracker()

    # -------------------- BACKEND INIT --------------------
    if args.backend == "phi":
        backend = PhiBackend(model=args.model)
    elif args.backend == "phi_reasoning":
        backend = PhiReasoningBackend(model=args.model)
    elif args.backend == "llama":
        backend = LlamaBackend(model=args.model)
    elif args.backend == "qwen":
        backend = QwenBackend(model=args.model)
    elif args.backend == "qwen3":
        backend = Qwen3Backend(model=args.model)
    elif args.backend == "qwen3_thinking":
        backend = Qwen3ThinkingBackend(model=args.model)
    elif args.backend == "qwq":
        backend = QwQBackend(model=args.model)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # -------------------- PATHS --------------------
    file_title = os.path.splitext(os.path.basename(args.file))[0]
    outdir = os.path.join(args.output_dir, file_title)
    os.makedirs(outdir, exist_ok=True)

    metapath   = os.path.join(outdir, args.sheet + "_meta.csv")
    preds_path = os.path.join(outdir, args.sheet + "_predictions.csv")
    rats_path  = os.path.join(outdir, args.sheet + "_rationales.csv")
    conf_path  = os.path.join(outdir, args.sheet + "_confidences.csv")

    if not os.path.exists(metapath):
        with open(metapath, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "idx", "sentence", "use_context_requested", "has_usable_context",
                "requested_context_mode", "arm_used", "context_included",
                "parse_status", "missing_key_count", "missing_keys",
            ])

    def writemeta(idx, sentence, usable, arm_used, context_included,
                  parse_status, missing_count, missing_keys):
        with open(metapath, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                idx, sentence,
                int(args.context), int(usable),
                args.context_mode if args.context else "",
                arm_used, int(bool(context_included)),
                parse_status, missing_count, missing_keys,
            ])

    # -------------------- LOAD DATA --------------------
    if args.gold:
        gold_path = args.gold
        print(f"Loading sentences from gold file: {gold_path}")
        golddf = pd.read_csv(gold_path) if gold_path.endswith('.csv') else pd.read_excel(gold_path)
    else:
        sheets = pd.read_excel(args.file, sheet_name=None)
        if "Gold" not in sheets:
            raise ValueError(f"Excel file must contain a 'Gold' sheet (or use --gold). Found: {list(sheets.keys())}")
        golddf = sheets["Gold"]

    if "sentence" not in golddf.columns:
        raise ValueError(f"Gold data must have 'sentence' column. Found: {list(golddf.columns)}")

    golddf = golddf.dropna(subset=["sentence"]).reset_index(drop=True)

    if args.context and "idx" not in golddf.columns:
        print("WARNING: Context requested but no 'idx' column found. Assuming row order = discourse order.")

    eval_sentences = golddf["sentence"].dropna().tolist()
    print(f"Sentences to evaluate: {len(eval_sentences)}")

    CURRENT_FEATURES   = EXTENDED_FEATURES if args.extended else MASIS_FEATURES
    BASE_FEATURE_BLOCK = NEW_FEATURE_BLOCK if args.extended else MASIS_FEATURE_BLOCK
    include_examples_in_block = args.block_examples

    # -------------------- RESUME SUPPORT --------------------
    def get_resume_idxs(preds_path, eval_sentences):
        if not os.path.exists(preds_path):
            return set()
        try:
            existing_df = pd.read_csv(preds_path)
            if len(existing_df) == 0:
                return set()
            if "idx" in existing_df.columns:
                last_row = existing_df.iloc[-1]
                missing_pct = last_row.isna().sum() / len(CURRENT_FEATURES)
                if missing_pct > 0.1:
                    print(f"WARNING: Last row incomplete ({missing_pct:.1%} missing). Re-processing.")
                    return set(existing_df['idx'].tolist()[:-1])
                completed = set(existing_df['idx'].tolist())
                print(f"INFO: Resume: {len(completed)} sentences already done.")
                return completed
            if "sentence" in existing_df.columns:
                num_rows = len(existing_df)
                if num_rows > len(eval_sentences):
                    return set()
                print(f"INFO: Resuming from row {num_rows} (sentence-match).")
                return set(range(num_rows))
            return set()
        except Exception as e:
            print(f"ERROR: Resume check failed: {e}. Starting fresh.")
            return set()

    existing_done_idxs = get_resume_idxs(preds_path, eval_sentences)

    preds_header = ["idx", "sentence"] + CURRENT_FEATURES
    rats_header  = ["idx", "sentence"] + CURRENT_FEATURES
    conf_header  = ["idx", "sentence"] + CURRENT_FEATURES

    if not os.path.exists(preds_path):
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(preds_header)
    if not os.path.exists(rats_path):
        with open(rats_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(rats_header)
    if not os.path.exists(conf_path):
        with open(conf_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(conf_header)

    # -------------------- COUNTERS --------------------
    usable_ctx_count = 0
    dump_counter = 0

    start_time = time.time()
    print(f"\nStart: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Backend: {args.backend} | Model: {args.model}")
    print(f"Output format: {args.output_format} | Instruction: {args.instruction_type}")

    # ==================== MAIN LOOP ====================
    for idx, sentence in enumerate(tqdm(eval_sentences, desc="Annotating")):
        if idx in existing_done_idxs:
            continue

        left = right = None
        usable = False

        if args.context:
            if args.context_mode == "wide":
                left_sents  = [golddf.loc[i, "sentence"] for i in range(max(0, idx - 5), idx)]
                right_sents = [golddf.loc[i, "sentence"] for i in range(idx + 1, min(len(golddf), idx + 6))]
                left  = "\n".join(left_sents)  if left_sents  else None
                right = "\n".join(right_sents) if right_sents else None
            else:
                if idx > 0:
                    left = golddf.loc[idx - 1, "sentence"]
                if idx < len(golddf) - 1:
                    right = golddf.loc[idx + 1, "sentence"]
            usable = has_usable_context(left, right)
            if usable:
                usable_ctx_count += 1

        context_included = bool(args.context and usable)

        raw, arm_used = query_model(
            backend,
            tracker,
            sentence,
            features=CURRENT_FEATURES,
            base_feature_block=BASE_FEATURE_BLOCK,
            instruction_type=args.instruction_type,
            include_examples_in_block=include_examples_in_block,
            output_format=args.output_format,
            use_context=args.context,
            left_context=left,
            right_context=right,
            context_mode=args.context_mode,
            dialect_legitimacy=args.dialect_legitimacy,
            dump_prompt=args.dump_prompt,
            dump_prompt_path=args.dump_prompt_path,
            dump_counter=dump_counter,
            dump_first_n=args.dump_first_n,
            sentence_idx=idx,
        )

        if args.dump_prompt and (args.dump_first_n == 0 or dump_counter < args.dump_first_n):
            dump_counter += 1

        print(f"\n{'─' * 60}")
        print(f"IDX {idx} | {sentence[:80]}...")
        print(f"{'─' * 60}")
        print(raw if raw else "(EMPTY RESPONSE)")
        print(f"{'─' * 60}\n")

        if not raw:
            writemeta(idx, sentence, usable, arm_used, context_included, "EMPTYRESPONSE", "", "")
            continue

        try:
            vals, rats, missing = parse_output(raw, CURRENT_FEATURES, output_format=args.output_format)
            parse_status  = "OK"
            missing_count = len(missing)
            missing_keys  = ",".join(missing)
            confidences   = extract_feature_confidences(backend, CURRENT_FEATURES, vals)

        except Exception as exc:
            print(f"Parse error at idx {idx}: {exc}")
            parse_status  = "PARSEFAIL"
            missing_count = ""
            missing_keys  = ""
            vals          = {f: None for f in CURRENT_FEATURES}
            rats          = {f: ""   for f in CURRENT_FEATURES}
            confidences   = {}

        if isinstance(missing_count, int) and missing_count > 5:
            print(f"WARNING: {missing_count} missing features at idx {idx}: {missing_keys}")

        writemeta(idx, sentence, usable, arm_used, context_included,
                  parse_status, missing_count, missing_keys)

        with open(preds_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [idx, sentence] + [vals.get(feat) for feat in CURRENT_FEATURES]
            )

        clean_rats = []
        for feat in CURRENT_FEATURES:
            r = rats.get(feat, "") or ""
            if not isinstance(r, str):
                r = str(r)
            r = r.replace("\n", " ").replace("\r", "")
            clean_rats.append(r)

        with open(rats_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f, quoting=csv.QUOTE_ALL).writerow(
                [idx, sentence] + clean_rats
            )

        if confidences:
            with open(conf_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [idx, sentence] + [confidences.get(feat, "") for feat in CURRENT_FEATURES]
                )

    # -------------------- FINAL SUMMARY --------------------
    print_final_usage_summary(tracker)
    if args.context:
        print(f"Sentences with usable context: {usable_ctx_count} / {len(eval_sentences)}")

    end_time = time.time()
    print(f"End:     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")


if __name__ == "__main__":
    main()