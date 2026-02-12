import os
import pandas as pd
import csv
import json
import torch
import re
import time
from tqdm import tqdm
import datetime
from openai import OpenAI
import tiktoken
import argparse
import math
from transformers import pipeline as hf_pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Dict, Any

"""
nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zs_ctx2_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
   --model gemini-2.5-flash \
   --backend gemini \
    --sheet GEMINI_ZS_CTX2_legit_json \
    --instruction_type zero_shot \
    --extended \
    --dialect_legitimacy \
    --context \
    --context_mode two_turn \
    --dump_prompt \
    --output_dir Decoder-Only/Gemini/data"

nlprun -q jag -p standard -r 40G -c 2 \
  -n gemini_zs_ctx_leg \
  -o Decoder-Only/Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
   --model gemini-2.5-flash \
   --backend gemini \
    --sheet GEMINI_ZS_CTX_legit \
    --instruction_type zero_shot \
    --extended \
    --dialect_legitimacy \
    --context \
    --dump_prompt \
    --output_dir Decoder-Only/Gemini/data"

nlprun -g 1 -q sphinx -p standard -r 100G -c 4 \
  -n phi4_gen_zs_ctx_leg \
  -o Decoder-Only/Phi-4/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python Decoder-Only/multi_prompt_configs.py \
    --file Datasets/FullTest_Final.xlsx \
   --model microsoft/phi-4  \
   --backend phi \
    --sheet PHI4_ZS_CTX_legit_json \
    --instruction_type zero_shot \
    --extended \
    --dialect_legitimacy \
    --context \
    --dump_prompt \
    --output_format json \
    --output_dir Decoder-Only/Phi-4/data"

nlprun -q jag -p standard -r 40G -c 2 \
  -n gem_ZS_noCTX_nolegit_labels \
  -o Gemini/slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/Dissertation/Decoder-Only && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate cgedit && \
   python multi_prompt_configs.py \
    --file FullTest_Final.xlsx \
   --model gemini-2.5-flash \
   --backend gemini \
    --sheet GEMINI_ZS_noCTX_nolegit_labels \
    --instruction_type zero_shot \
    --extended \
    --dump_prompt \
    --output_dir Gemini/data \
    --labels_only"
"""

"""
RESTRUCTURED PROMPT CONDITIONS

Old naming:
- zero_shot, icl, zero_shot_cot, few_shot_cot

New naming:
- zero_shot, few_shot, zero_shot_cot, few_shot_cot

Key changes:
1. 'icl' renamed to 'few_shot' (clearer)
2. Two separate ICL blocks: labels-only vs CoT
3. --labels_only flag removed (Markdown is always CoT; use --output_format json for compact)
4. CoT vs non-CoT differs in DEPTH of reasoning requested, not presence/absence

Example commands:

# Zero-shot with brief reasoning
nlprun ... python script.py --instruction_type zero_shot --output_format markdown

# Zero-shot with detailed step-by-step reasoning
nlprun ... python script.py --instruction_type zero_shot_cot --output_format markdown

# Few-shot with brief reasoning (includes 3 labels-only examples)
nlprun ... python script.py --instruction_type few_shot --output_format markdown

# Few-shot with detailed reasoning (includes 3 CoT examples with rationales)
nlprun ... python script.py --instruction_type few_shot_cot --output_format markdown

# Compact JSON output (no reasoning, just labels)
nlprun ... python script.py --instruction_type zero_shot --output_format json
"""

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
"""
CONFIDENCE EXTRACTION SUPPORT BY BACKEND:

 OpenAI (GPT-4, GPT-4o):
   - Feature-specific confidence via token-level logprobs
   - High accuracy: directly tied to model's uncertainty per label
   - Recommended for research requiring confidence scores

 Phi-4 / Qwen (HuggingFace local):
   - Global proxy: average max probability across all tokens
   - Same confidence assigned to ALL features (not feature-specific)
   - Use as rough indicator of overall model certainty

 Gemini (Google API):
   - No confidence support (API doesn't expose logprobs)
   - Returns empty dict from extract_feature_confidences()
   - Use OpenAI backend if confidence is required
"""

@dataclass
class LLMBackend:
    """
    Abstract base. Each backend owns its own tokenizer/encoder.
    """
    name: str
    model: str

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        raise NotImplementedError

    def count_output_tokens(self, text: str) -> int:
        raise NotImplementedError

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        raise NotImplementedError  # ← Updated signature


class OpenAIBackend(LLMBackend):
    """
    OpenAI API backend (GPT-4, GPT-4o, etc.).

    CONFIDENCE SUPPORT: Feature-specific
    - Extracts token-level logprobs for each binary label (0/1)
    - Matches logprobs to features using regex on preceding tokens
    - High accuracy: confidence directly tied to model's uncertainty
    """
    def __init__(self, model: str):
        super().__init__(name="openai", model=model)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")
        self._last_confidence_data = None
    
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        text = "\n".join(m["content"] for m in messages)
        return len(self._enc.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        # Set max_tokens based on instruction type
        if "cot" in instruction_type:
            max_tokens = 16000
        else:
            max_tokens = 4000
        
        resp = self.client.chat.completions.create(
            model=self.model, 
            messages=messages,
            temperature=0.1,  # Add explicit temperature
            max_tokens=max_tokens,  # Add explicit max_tokens
            logprobs=True,  # ← Add this
            top_logprobs=2,  # ← Add this (returns top 2 token probabilities)
        )

        # Store logprobs for later analysis (optional)
        if hasattr(resp.choices[0], 'logprobs') and resp.choices[0].logprobs:
            self._last_confidence_data = {
                'type': 'logprobs',
                'data': resp.choices[0].logprobs
            }

        return resp.choices[0].message.content


class QwenBackend(LLMBackend):
    """
    HuggingFace Qwen model backend (local inference).

    CONFIDENCE SUPPORT: Global proxy only
    - Extracts average max probability across all generated tokens
    - Same confidence assigned to ALL features (not feature-specific)
    - Use as rough proxy for overall model uncertainty
    """
    def __init__(self, model: str):
        super().__init__(name="qwen", model=model)
        print(f"Loading Qwen from {model}...")

        self.model_id = model  # ← Store string ID before overwriting
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device  # ← Store device

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
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set max_new_tokens based on instruction type
        if "cot" in instruction_type:
            max_tokens = 16000
        else:
            max_tokens = 2000

        # Use model.generate() directly to get scores
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            return_dict_in_generate=True,  # ← Now works!
            output_scores=True,  # ← Now works!
        )
        


        
        # Decode generated text (skip input tokens)
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract scores for confidence
        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text  # Before markdown stripping
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

class GeminiBackend(LLMBackend):
    """
    Google Gemini API backend (Gemini 2.0/2.5 Flash/Pro).

    CONFIDENCE SUPPORT: Not supported
    - Gemini API does not expose logprobs or token probabilities
    - Confidence extraction will return empty dict
    - Use OpenAI backend if confidence scores are required
    """
    def __init__(self, model: str):
        super().__init__(name="gemini", model=model)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.cache = None
        self.cached_model_client = None
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._last_confidence_data = None

    def count_tokens(self, messages):
        # Use Gemini's native token counting
        text = "\n".join(m["content"] for m in messages)
        try:
            # Gemini has a built-in token counter
            model = genai.GenerativeModel(self.model)
            return model.count_tokens(text).total_tokens
        except:
            # Fallback to tiktoken approximation
            return len(self._enc.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def create_cache(self, system_instruction: str, model_name: str, ttl_minutes: int = 60):
        """Creates a cached content object on Gemini servers."""
        print("Creating Gemini Context Cache...")
        try:
            from google.generativeai import caching
            self.cache = caching.CachedContent.create(
                model=model_name,
                display_name="aae_annotation_cache",
                system_instruction=system_instruction,
                ttl=datetime.timedelta(minutes=ttl_minutes),
            )
            self.cached_model_client = genai.GenerativeModel.from_cached_content(self.cache)
            print(f"Cache created! Name: {self.cache.name}")
            return True
        except ImportError:
            print("WARNING: google.generativeai.caching not found. pip install -U google-generativeai")
            return False
        except Exception as e:
            print(f"WARNING: Failed to create cache: {e}")
            return False
    

    def call(self, messages: List[Dict[str, str]], instruction_type: str = "zero_shot") -> str:
        # Set max_output_tokens based on instruction type
        if "cot" in instruction_type:
            max_tokens = 16000
        else:
            max_tokens = 4000
        
        generation_config = {
            "temperature": 0.1,
            "max_output_tokens": max_tokens,
            # REMOVED: "candidate_count": 3,  ← This parameter is NOT supported by Gemini API
        }
        
        try:
            client = self.cached_model_client or self.client
            user_turns = [m for m in messages if m["role"] == "user"]

            if len(user_turns) <= 1:
                # Single-turn: simple generation
                content = user_turns[0]["content"] if user_turns else ""
                # Apply system instruction if present and not using cached model
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
                if system_msg and not self.cached_model_client:
                    model_with_system = genai.GenerativeModel(
                        model_name=self.model,
                        system_instruction=system_msg
                    )
                    resp = model_with_system.generate_content(
                        content,
                        generation_config=generation_config
                    )
                else:
                    resp = client.generate_content(
                        content,
                        generation_config=generation_config
                    )
                # REMOVED: Candidate extraction code (candidates not supported)
            else:
                # Multi-turn: use chat with history
                # Extract system message (if not cached)
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
                
                # Build history from user messages
                history = []
                for m in messages:
                    if m["role"] == "user":
                        history.append({"role": "user", "parts": [m["content"]]})
                
                # Last message is the query
                last_content = history.pop()["parts"][0]
                
                # If we have a system message and NOT using cache, create model with system instruction
                if system_msg and not self.cached_model_client:
                    model_with_system = genai.GenerativeModel(
                        model_name=self.model,
                        system_instruction=system_msg
                    )
                    chat = model_with_system.start_chat(history=history)
                else:
                    # Using cached model (system already on server) or no system message
                    chat = client.start_chat(history=history)

                resp = chat.send_message(last_content, generation_config=generation_config)

            return resp.text
        except Exception as e:
            raise e
        

class PhiBackend(LLMBackend):
    """
    HuggingFace Phi-4 model backend (local inference).

    CONFIDENCE SUPPORT: Global proxy only
    - Extracts average max probability across all generated tokens
    - Same confidence assigned to ALL features (not feature-specific)
    - Use as rough proxy for overall model uncertainty
    """
    def __init__(self, model: str):
        super().__init__(name="phi", model=model)
        print(f"Loading Phi-4 from {model}...")

        self.model_id = model  # ← Store string ID before overwriting
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="auto",
            device_map="auto",
        )
        self.device = self.model.device  # ← Add this
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
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set max_new_tokens based on instruction type
        if "cot" in instruction_type:
            max_tokens = 16000
        else:
            max_tokens = 2000

        # Use model.generate() directly to get scores
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            return_dict_in_generate=True,  # ← Now works!
            output_scores=True,  # ← Now works!
        )
        

        
        # Decode generated text (skip input tokens)
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract scores for confidence
        if hasattr(outputs, 'scores') and outputs.scores:
            self._last_confidence_data = {
                'type': 'hf_scores',
                'tokenizer': self.tokenizer,
                'generated_ids': generated_ids,
                'scores': outputs.scores,
                'generated_text': generated_text  # Before markdown stripping
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

    # Pull the Analysis block as a single global rationale
    analysis_match = re.search(r"### Analysis\s*(.*?)\s*### Results", text, re.DOTALL)
    global_rationale = analysis_match.group(1).strip() if analysis_match else "No analysis found."

    # Extract binary labels from the Results block
    pattern = r"[-\*]\s*([\w-]+):\s*(0|1)"
    for key, val in re.findall(pattern, text):
        if key in expected_features:
            vals[key] = int(val)
            rats[key] = f"See analysis: {global_rationale[:200]}..."

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


def parse_output(raw_str: str, features: list, output_format: str = "markdown") -> tuple:
    """
    Unified output parser.
    Tries the format that was requested first, then falls back to the other.
    Always returns (vals, rats, missing).
    """
    vals, rats, missing = {}, {}, list(features)

    if output_format == "markdown":
        vals, rats, missing = extract_results(raw_str, features)
        # Fallback: model may have returned JSON despite being asked for Markdown
        if missing:
            data = extract_json_robust(raw_str)
            if data:
                for f in list(missing):
                    if f in data:
                        v = data[f]
                        vals[f] = v if isinstance(v, int) else int(v.get("value", 0))
                        rats[f] = str(v.get("rationale", "")) if isinstance(v, dict) else ""
                        missing.remove(f)
    else:
        data = extract_json_robust(raw_str)
        # Extract reasoning text before the JSON block as global rationale
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
        # Fallback: model may have returned Markdown despite being asked for JSON
        if missing:
            md_vals, md_rats, md_missing = extract_results(raw_str, features)
            for f in list(missing):
                if f in md_vals:
                    vals[f] = md_vals[f]
                    rats[f] = md_rats.get(f, "")
                    missing.remove(f)

    return vals, rats, missing


def extract_feature_confidences(backend: LLMBackend, feature_names: list, parsed_labels: dict) -> dict:
    """
    Extract confidence scores from any backend's last call.

    Backend support:
    - OpenAI: Token-level logprobs (feature-specific)
    - Phi/Qwen (HuggingFace): Average max probability (global proxy)
    - Gemini: Not supported (returns empty dict)

    Returns dict: {feature_name: confidence_score}
    where confidence_score is in [0, 1], higher = more confident.
    """
    if not hasattr(backend, '_last_confidence_data') or not backend._last_confidence_data:
        return {}

    conf_data = backend._last_confidence_data
    conf_type = conf_data.get('type')

    if not conf_type:
        return {}

    confidences = {}

    if conf_type == 'logprobs':
        # OpenAI logprobs - feature-specific confidence extraction
        data = conf_data.get('data')
        try:
            if not data or not hasattr(data, 'content') or not data.content:
                print("Warning: OpenAI logprobs data missing 'content' attribute")
                return {}

            tokens = data.content

            for i, token_data in enumerate(tokens):
                if not hasattr(token_data, 'token') or not hasattr(token_data, 'logprob'):
                    continue

                token = token_data.token.strip()
                if token not in ["0", "1"]:
                    continue

                # Build lookback context (previous 15 tokens)
                lookback_text = ""
                for j in range(max(0, i-15), i):
                    if hasattr(tokens[j], 'token'):
                        lookback_text += tokens[j].token

                # Find matching feature
                matched = False
                for feat in feature_names:
                    pattern = rf'-\s*{re.escape(feat)}\s*:\s*$'
                    if re.search(pattern, lookback_text):
                        prob = max(math.exp(token_data.logprob), 1e-10)  # Floor at 1e-10
                        if feat not in confidences:  # Only set if not already set
                            confidences[feat] = prob
                        matched = True
                        break  # Assume only one feature per token

                if not matched:
                    # Uncomment for debugging:
                    # print(f"DEBUG: No feature match for token '{token}' at position {i}")
                    pass

        except AttributeError as e:
            print(f"Warning: OpenAI logprobs structure error: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error extracting OpenAI logprobs: {e}")

    elif conf_type == 'hf_scores':
        # HuggingFace scores (Phi/Qwen) - global proxy confidence
        scores = conf_data.get('scores')
        try:
            if not isinstance(scores, (list, tuple)):
                print(f"Warning: HF scores data is not a list/tuple (got {type(scores)})")
                return {}

            # Get average max probability across all generated tokens
            max_probs = []
            for score_tensor in scores:
                if not hasattr(score_tensor, 'shape'):
                    continue
                probs = torch.nn.functional.softmax(score_tensor[0], dim=-1)
                max_prob = torch.max(probs).item()
                max_probs.append(max_prob)

            # Calculate average confidence
            avg_confidence = sum(max_probs) / len(max_probs) if max_probs else 0.5

            # Assign same confidence to all features (proxy measure)
            # This is a global confidence, not feature-specific
            for feat in feature_names:
                confidences[feat] = avg_confidence

        except Exception as e:
            print(f"Warning: Could not extract HF scores: {e}")

    else:
        # Unknown confidence type
        print(f"Warning: Unknown confidence type '{conf_type}'")
    
    # elif conf_type == 'candidates':
    #     # Gemini multiple candidates (variance-based confidence)
    #     # NOTE: This branch is currently unused because Gemini doesn't support candidate_count
    #     # Keeping for potential future use if Gemini adds this feature
    #     candidate_texts = data
        
    #     # Parse each candidate
    #     all_parses = []
    #     for candidate_text in candidate_texts:
    #         try:
    #             vals, _, _ = parse_output(candidate_text, feature_names, output_format="markdown")
    #             all_parses.append(vals)
    #         except:
    #             continue
        
    #     if len(all_parses) >= 2:
    #         # For each feature, compute agreement rate
    #         for feat in feature_names:
    #             # Guard against missing or None labels
    #             if feat in parsed_labels and parsed_labels.get(feat) is not None:
    #                 # Count how many candidates agree with the final label
    #                 final_label = parsed_labels[feat]
    #                 agreements = sum(1 for p in all_parses if p.get(feat) == final_label)
    #                 confidence = agreements / len(all_parses)
    #                 confidences[feat] = confidence
    
    return confidences




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
# These are kept here as module-level constants.
# They are passed into build_system_msg (system role) rather than the user message,
# so they are static, cacheable, and don't re-inflate every user turn.

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

Note: If there is a recoverable subject and predicate slot that would host BE in SAE, mark 1 even if the utterance is short or informal.

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

### Rule 15: zero-pl-s
**IF** a noun that clearly has plural reference (from a quantifier, determiner, or context) surfaces without SAE plural -s AND the plural reading is local to the noun phrase, **THEN** label is 1.

+ Example: "She got them dog."
  * Label is 1
  * Explanation: Plural demonstrative 'them' + bare 'dog'

– Example: "A dogs."
  * Label is 0
  * Explanation: Article–noun mismatch, not AAE plural pattern

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

### Rule 23: past-tense-swap
**IF** an overtly non-SAE tense form is used as the main tense carrier of a clause AND the clause has clear simple-past or perfect/pluperfect reference, **THEN** label is 1.

Mark 1 when:
* A past participle is used as simple past (e.g., 'seen', 'done' for 'saw', 'did'), OR
* A regularized past is used where SAE requires irregular (e.g., 'throwed' for 'threw'), OR
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

Note: This feature never applies to bare stems (those are verb-stem). If the verb has no overt tense morphology, mark 0 for past-tense-swap.

### Rule 24: zero-rel-pronoun
**IF** a finite clause modifies a noun and functions as a subject relative AND there is NO overt relative pronoun ('who', 'that', 'which') in subject position, **THEN** label is 1.

+ Example: "There are many mothers don't know their children."
  * Label is 1
  * Explanation: Clause modifying 'mothers' without 'who'

– Example: "I think he left."
  * Label is 0
  * Explanation: That-deletion in complement clause, not subject relative

### Rule 25: preterite-had
**IF** 'had' plus a past verb is used to express a simple past event (with no clear 'past-before-past' meaning) AND there is no later past event anchoring a pluperfect reading, **THEN** label is 1.

Often appears with regularized/AAE-style past forms (had went, had ran).

+ Example: "The alarm next door had went off a few minutes ago."
  * Label is 1
  * Explanation: Simple past meaning; no later reference event

– Example: "They had seen the movie before we arrived."
  * Label is 0
  * Explanation: True pluperfect (past-before-past)

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



# ==================== ICL BLOCKS ====================

# For 'few_shot' condition: labels only, no rationales
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

Use these as patterns for complete sentence annotation.
Do NOT reuse these sentences when analyzing the new target utterance.
"""


# For 'few_shot_cot' condition: labels + detailed rationales
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

Use these as patterns: for each feature, identify the exact grammatical evidence and apply the decision rule.
Do NOT reuse these sentences when analyzing the new target utterance.
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
        lines.append(f"PREV SENTENCE (context): {left_context}")
    if right_context:
        lines.append(f"NEXT SENTENCE (context): {right_context}")
    return "\n".join(lines)

# ==================== PROMPT BUILDERS ====================
#
# SYSTEM / USER SPLIT RATIONALE
# ─────────────────────────────
# SYSTEM message (static, cacheable):
#   - Persona and legitimacy framing
#   - Annotation procedure (clause analysis, feature eval, etc.)
#   - Evaluation constraints
#   - The full feature rule block (3 000+ tokens — expensive to re-send)
#   - ICL examples (when used — also static across sentences)
#
# USER message (per-sentence, cheap):
#   - Optional context block (PREV/NEXT sentences)
#   - Target utterance
#   - Feature key list
#   - Output format instructions (markdown or JSON)
#
# This split means the expensive rule block is sent once (or cached on Gemini/OpenAI),
# and each per-sentence call only adds the small user turn.


def build_system_msg(
    instruction_type: str,
    dialect_legitimacy: bool,
    use_context: bool,
    base_feature_block: str,
    include_examples_in_block: bool,
) -> dict:
    """
    Build the full system message.
    Includes: framing, procedure, constraints, feature rules, and (optionally) ICL examples.
    All static content lives here so it can be cached.
    """

    # --- Framing ---
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

    # --- Core procedure ---
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

    # --- Reasoning depth (CoT vs brief) ---
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

    # --- Feature rules ---
    effective_include_examples = include_examples_in_block
    if instruction_type in ["zero_shot", "zero_shot_cot"]:
        effective_include_examples = False
    feature_rules = base_feature_block if effective_include_examples else strip_examples_from_block(base_feature_block)

    # --- ICL examples (conditional on instruction type) ---
    icl_block = ""
    if instruction_type == "few_shot":
        icl_block = ICL_LABELS_ONLY_BLOCK
    elif instruction_type == "few_shot_cot":
        icl_block = ICL_COT_BLOCK
    # zero_shot and zero_shot_cot get no ICL block

    content = "".join([
        intro,
        procedure,
        constraints,
        ctx_instructions,
        cot_instruction,
        feature_rules,
        icl_block,
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
    Build the per-sentence user message content.
    Kept deliberately small: context (optional) + utterance + feature list + output instructions.
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
            "After your analysis, output ALL features as a bulleted list:\n"
            "- feature-name: 1\n"
            "- feature-name: 0\n\n"
            "Every feature in the list below must appear exactly once.\n"
        )
    else:
        # JSON mode: reason first, then output JSON
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

    return (
        f"{context_section}"
        f"### TARGET UTTERANCE ###\n"
        f'"{utterance}"\n\n'
        f"### FEATURES TO LABEL ###\n"
        f"[{feature_list_str}]\n\n"
        f"{output_instructions}"
    )

def build_messages(
    utterance: str,
    features: list,
    base_feature_block: str,
    instruction_type: str,
    include_examples_in_block: bool,
    output_format: str = "markdown",
    use_context: bool = False,
    left_context: str | None = None,
    right_context: str | None = None,
    context_mode: str = "single_turn",
    dialect_legitimacy: bool = False,
) -> tuple[list[dict], str]:
    """
    Assembles the full messages list for a single sentence.

    Returns (messages, arm_used) where arm_used is "single_turn" or "two_turn".

    Message structure:
        [system_msg]                           — always present
        [context_msg]                          — only in two_turn mode
        [user_msg]                             — always present
    """

    system_msg = build_system_msg(
        instruction_type=instruction_type,
        dialect_legitimacy=dialect_legitimacy,
        use_context=use_context,
        base_feature_block=base_feature_block,
        include_examples_in_block=include_examples_in_block,
    )

    # Resolve context
    context_block = None
    if use_context:
        cb = format_context_block(left_context, right_context)
        if cb.strip():
            context_block = cb

    # Two-turn mode: context in its own prior user message
    if use_context and context_block and context_mode == "two_turn":
        context_msg = {
            "role": "user",
            "content": (
                "### CONTEXT ###\n"
                "(Use ONLY to resolve subject reference and tense, NOT to infer events)\n\n"  # ← Reminder here
                "CONTEXT (do NOT analyze yet; this is NOT the target utterance).\n\n"
                f"{context_block}"
            ),
        }
        user_content = build_user_msg(
            utterance=utterance,
            features=features,
            output_format=output_format,
            instruction_type=instruction_type,
            context_block=None,  # ← No reminder in TARGET message
        )
        user_msg = {"role": "user", "content": user_content}
        return [system_msg, context_msg, user_msg], "two_turn"

    # Single-turn (default): context embedded in user message (reminder goes here)
    user_content = build_user_msg(
        utterance=utterance,
        features=features,
        output_format=output_format,
        instruction_type=instruction_type,
        context_block=context_block,  # ← Reminder included via build_user_msg
    )
    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg], "single_turn"

# ==================== USAGE SUMMARY ====================

def print_final_usage_summary(tracker: TokenTracker):
    total_tokens = tracker.total_input_tokens + tracker.total_output_tokens
    print(f"Total API Calls:     {tracker.api_call_count}")
    print(f"Input Tokens:        {tracker.total_input_tokens}")
    print(f"Output Tokens:       {tracker.total_output_tokens}")
    print(f"Total Tokens:        {total_tokens}")

# ==================== MODEL QUERY ====================


def query_model(
    backend: LLMBackend,
    tracker: TokenTracker,
    sentence: str,
    features: list,
    base_feature_block: str,
    instruction_type: str,
    include_examples_in_block: bool,
    output_format: str = "markdown",
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
    max_retries: int = 15,
    base_delay: int = 3,
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

    # For Gemini with caching: keep full messages for the dump, strip system only for the API call
    full_messages_for_dump = messages  # Always dump the complete messages
    if isinstance(backend, GeminiBackend) and backend.cached_model_client is not None:
        messages = [m for m in messages if m["role"] != "system"]

    # Enhanced prompt dump with context verification
    should_dump = dump_prompt and (dump_first_n == 0 or dump_counter < dump_first_n)

    if should_dump:
        print("\n" + "="*80)
        print(f"PROMPT DUMP #{dump_counter + 1} (Sentence idx: {sentence_idx})")
        print("="*80)

        # Context verification section
        print("\n📍 CONTEXT VERIFICATION:")
        print(f"   use_context flag:     {use_context}")
        print(f"   context_mode:         {context_mode}")
        print(f"   arm_used:             {arm_used}")
        print(f"   left_context:         {'Present' if left_context and str(left_context).strip() else 'None/Empty'}")
        print(f"   right_context:        {'Present' if right_context and str(right_context).strip() else 'None/Empty'}")

        if left_context and str(left_context).strip():
            print(f"\n   LEFT (previous):  \"{str(left_context).strip()[:100]}{'...' if len(str(left_context).strip()) > 100 else ''}\"")
        if right_context and str(right_context).strip():
            print(f"   RIGHT (next):     \"{str(right_context).strip()[:100]}{'...' if len(str(right_context).strip()) > 100 else ''}\"")

        print(f"\n   TARGET SENTENCE:  \"{sentence[:100]}{'...' if len(sentence) > 100 else ''}\"")

        # Message structure (always show full messages including system)
        print(f"\nMESSAGE STRUCTURE:")
        print(f"   Total messages:       {len(full_messages_for_dump)}")
        for i, msg in enumerate(full_messages_for_dump):
            role = msg['role']
            content_preview = msg['content'][:150].replace('\n', ' ')
            print(f"   [{i}] {role:8s}:  {content_preview}...")

        # Full JSON dump
        print(f"\nFULL PROMPT JSON:")

        # Get model name string (HF backends store it in model_id)
        model_name = getattr(backend, 'model_id', backend.model)
        if not isinstance(model_name, str):
            model_name = backend.name  # Fallback to backend name

        payload = {
            "backend": backend.name,
            "model": model_name,
            "sentence_idx": sentence_idx,
            "context_info": {
                "use_context": use_context,
                "context_mode": context_mode,
                "arm_used": arm_used,
                "left_context": left_context,
                "right_context": right_context,
            },
            "messages": full_messages_for_dump
        }
        if isinstance(backend, GeminiBackend) and backend.cached_model_client is not None:
            payload["note"] = "System message is cached on Gemini server (not sent per-request)"
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        print("\n" + "="*80)
        print("END PROMPT DUMP")
        print("="*80 + "\n")

        # Write to file if path specified
        if dump_prompt_path:
            # Append dump number to filename if dumping multiple
            if dump_first_n != 1:
                base, ext = os.path.splitext(dump_prompt_path)
                output_path = f"{base}_{dump_counter + 1}{ext}"
            else:
                output_path = dump_prompt_path

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"💾 Prompt saved to: {output_path}\n")

    # Count input tokens BEFORE calling API
    input_tokens = backend.count_tokens(messages)
    
    # Context length check
    if "cot" in instruction_type:
        estimated_output = 4500
        context_limit = 32000
    else:
        estimated_output = 1000
        context_limit = 32000
    
    total_tokens = input_tokens + estimated_output
    
    if total_tokens > context_limit:
        print(f"WARNING: Estimated {total_tokens} tokens for sentence. "
              f"May exceed model context limit ({context_limit}).")

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            # Make API call
            output_text = backend.call(messages, instruction_type=instruction_type)

            # Count output tokens AFTER successful call
            output_tokens = backend.count_output_tokens(output_text)

            tracker.total_input_tokens += input_tokens
            tracker.total_output_tokens += output_tokens
            tracker.api_call_count += 1

            # Print summary
            print(
                f"API Call #{tracker.api_call_count} | "
                f"Input: {input_tokens} | Output: {output_tokens} | "
                f"Running total: {tracker.total_input_tokens + tracker.total_output_tokens}"
            )

            return output_text, arm_used

        except Exception as e:
            msg = str(e).lower()
            is_last_attempt = (attempt == max_retries - 1)

            # Classify error type
            if "429" in msg or "rate" in msg or "quota" in msg:
                # Rate limit error - exponential backoff
                wait_time = base_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

            elif "timeout" in msg or "timed out" in msg:
                # Timeout error - retry with backoff
                wait_time = base_delay * (2 ** (attempt // 2))  # Slower backoff
                print(f"Timeout error. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

            elif "503" in msg or "500" in msg or "502" in msg:
                # Server error - retry with moderate backoff
                wait_time = base_delay * (1.5 ** attempt)
                print(f"Server error ({msg[:50]}). Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

            else:
                # Unknown error - don't retry unless it's not the last attempt
                print(f"Error on sentence: {sentence[:40]}...")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)[:200]}")

                if not is_last_attempt:
                    wait_time = base_delay
                    print(f"Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return None, arm_used

    print(f"Failed after {max_retries} retries. Skipping sentence.")
    return None, arm_used


# ==================== MAIN ====================

def main():
    print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"DEBUG: torch.version.cuda        = {torch.version.cuda}")

    parser = argparse.ArgumentParser(description="Run LLM experiments for AAE feature annotation.")
    parser.add_argument("--file",             type=str, required=True,  help="Input Excel file path")
    parser.add_argument("--sheet",            type=str, required=True,  help="Sheet name for predictions")
    parser.add_argument("--extended",         action="store_true",      help="Use extended 27-feature set")
    parser.add_argument("--context",          action="store_true",      help="Include prev/next sentence context")
    parser.add_argument("--instruction_type", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "zero_shot_cot", "few_shot_cot"],
                        help="Instruction style: zero_shot (brief reasoning), few_shot (brief + examples), "
                             "zero_shot_cot (detailed reasoning), few_shot_cot (detailed + examples)")
    parser.add_argument("--block_examples",   action="store_true",
                        help="Keep example lines in feature block (non-zero-shot only)")
    parser.add_argument("--dialect_legitimacy", action="store_true",
                        help="Frame AAE as rule-governed and legitimate")
    parser.add_argument("--output_dir",       type=str, required=True,  help="Output directory")
    parser.add_argument("--dump_prompt",      action="store_true",      help="Print prompt (with context details)")
    parser.add_argument("--dump_prompt_path", type=str, default=None,   help="Write prompt JSON to this path")
    parser.add_argument("--dump_first_n",     type=int, default=1,      help="Dump first N prompts (default: 1, use 0 for all)")
    parser.add_argument("--context_mode",     type=str, default="single_turn",
                        choices=["single_turn", "two_turn"])
    parser.add_argument("--backend",          type=str, default="openai",
                        choices=["openai", "gemini", "phi", "qwen"])
    parser.add_argument("--model",            type=str, default="gpt-4o",
                        help="Model identifier (e.g. gpt-4o, gemini-2.0-flash-exp, microsoft/Phi-4)")
    parser.add_argument("--output_format",    type=str, default="markdown",
                        choices=["json", "markdown"],
                        help="'markdown' (Analysis + Results list) or 'json' (Analysis + JSON labels)")
    parser.add_argument("--rate_limit_delay", type=float, default=5.0,
                        help="Delay in seconds between API calls (0 for none)")


    args = parser.parse_args()
    tracker = TokenTracker()

    # -------------------- BACKEND INIT (single block) --------------------
    if args.backend == "openai":
        backend = OpenAIBackend(model=args.model)
    elif args.backend == "gemini":
        backend = GeminiBackend(model=args.model)
    elif args.backend == "phi":
        backend = PhiBackend(model=args.model)
    elif args.backend == "qwen":
        backend = QwenBackend(model=args.model)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # -------------------- PATHS --------------------
    file_title = os.path.splitext(os.path.basename(args.file))[0]
    outdir = os.path.join(args.output_dir, file_title)
    os.makedirs(outdir, exist_ok=True)

    metapath  = os.path.join(outdir, args.sheet + "_meta.csv")
    preds_path = os.path.join(outdir, args.sheet + "_predictions.csv")
    rats_path  = os.path.join(outdir, args.sheet + "_rationales.csv")
    conf_path  = os.path.join(outdir, args.sheet + "_confidences.csv")  # ← New file

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
    sheets = pd.read_excel(args.file, sheet_name=None)

    # Validate sheet exists
    if "Gold" not in sheets:
        raise ValueError(f"Excel file must contain a 'Gold' sheet. Found: {list(sheets.keys())}")

    golddf = sheets["Gold"]

    # Validate column exists
    if "sentence" not in golddf.columns:
        raise ValueError(f"'Gold' sheet must have 'sentence' column. Found: {list(golddf.columns)}")

    golddf = golddf.dropna(subset=["sentence"]).reset_index(drop=True)

    # Warn if context is requested but no ordering info
    if args.context and "idx" not in golddf.columns:
        print("WARNING: Context requested but no 'idx' column found. Assuming row order = discourse order.") 
        
    eval_sentences = golddf["sentence"].dropna().tolist()
    print(f"Sentences to evaluate: {len(eval_sentences)}")

    CURRENT_FEATURES  = EXTENDED_FEATURES if args.extended else MASIS_FEATURES
    BASE_FEATURE_BLOCK = NEW_FEATURE_BLOCK if args.extended else MASIS_FEATURE_BLOCK
    include_examples_in_block = args.block_examples

    # -------------------- RESUME SUPPORT --------------------
    def get_resume_idxs(preds_path, eval_sentences):
        """
        Strict resume logic with data corruption detection.
        Returns set of already-completed indices.
        """
        if not os.path.exists(preds_path):
            return set()

        try:
            existing_df = pd.read_csv(preds_path)

            if len(existing_df) == 0:
                print("INFO: Existing predictions file is empty. Starting fresh.")
                return set()

            # STRATEGY 1: Use 'idx' column if available (preferred)
            if "idx" in existing_df.columns:
                # Check last row completeness
                last_row = existing_df.iloc[-1]
                missing_in_last = last_row.isna().sum()
                missing_pct = missing_in_last / len(CURRENT_FEATURES)

                if missing_pct > 0.1:  # >10% missing
                    print(f"WARNING: Last row (idx {last_row.get('idx', '?')}) has {missing_pct:.1%} missing features.")
                    print(f"         This suggests the last annotation was interrupted.")
                    print(f"         Re-processing idx {last_row.get('idx', '?')} only.")
                    return set(existing_df['idx'].tolist()[:-1])  # Exclude last row

                completed_idxs = set(existing_df['idx'].tolist())
                print(f"INFO: Resume from predictions file. {len(completed_idxs)} sentences already completed.")

                # Validation: check for gaps or duplicates
                if len(completed_idxs) != len(existing_df):
                    duplicates = len(existing_df) - len(completed_idxs)
                    print(f"WARNING: Found {duplicates} duplicate idx values in predictions file.")

                max_idx = max(completed_idxs) if completed_idxs else -1
                expected_count = max_idx + 1
                if len(completed_idxs) != expected_count:
                    missing_count = expected_count - len(completed_idxs)
                    print(f"WARNING: {missing_count} missing idx values detected (gaps in sequence 0-{max_idx}).")
                    print(f"         This may indicate data corruption. Review predictions file manually.")

                return completed_idxs

            # STRATEGY 2: Fallback to sentence matching (less reliable)
            if "sentence" in existing_df.columns:
                print("WARNING: No 'idx' column found. Using sentence-based resume (less reliable).")
                num_rows = len(existing_df)

                # Verify alignment with current data
                if num_rows > len(eval_sentences):
                    print(f"ERROR: Predictions file has {num_rows} rows but input only has {len(eval_sentences)} sentences.")
                    print(f"       Data mismatch detected. Please verify input file hasn't changed.")
                    return set()  # Safer to restart

                # Check last sentence matches
                last_csv_sentence = str(existing_df.iloc[-1]["sentence"]).strip().lower()
                expected_sentence = str(eval_sentences[num_rows - 1]).strip().lower()

                if last_csv_sentence == expected_sentence:
                    print(f"INFO: Resume verified. Resuming from row {num_rows}.")
                    return set(range(num_rows))
                else:
                    print("ERROR: Sentence mismatch detected between predictions file and input data.")
                    print(f"       CSV row {num_rows-1}: '{last_csv_sentence[:50]}...'")
                    print(f"       Input sentence {num_rows-1}: '{expected_sentence[:50]}...'")
                    print(f"       Input file may have changed. Starting fresh to avoid corruption.")
                    return set()

            print("WARNING: Predictions file has neither 'idx' nor 'sentence' columns. Starting fresh.")
            return set()

        except pd.errors.EmptyDataError:
            print("INFO: Predictions file exists but is empty. Starting fresh.")
            return set()
        except Exception as e:
            print(f"ERROR: Resume check failed: {e}")
            print(f"       Starting fresh to avoid data corruption.")
            return set()

    existing_done_idxs = get_resume_idxs(preds_path, eval_sentences)

    preds_header = ["idx", "sentence"] + CURRENT_FEATURES
    rats_header  = ["idx", "sentence"] + CURRENT_FEATURES
    conf_header = ["idx", "sentence"] + CURRENT_FEATURES
    if not os.path.exists(preds_path):
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(preds_header)
    if not os.path.exists(rats_path):
        with open(rats_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(rats_header)
    # Create confidence CSV header
    if not os.path.exists(conf_path):
        with open(conf_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(conf_header)
    # -------------------- GEMINI CACHING --------------------
    if args.backend == "gemini":
        print("Initializing Gemini cache...")
        cache_start = time.time()
        dummy_msgs, _ = build_messages(
            utterance="DUMMY",
            features=CURRENT_FEATURES,
            base_feature_block=BASE_FEATURE_BLOCK,
            instruction_type=args.instruction_type,
            include_examples_in_block=include_examples_in_block,
            output_format=args.output_format,
            dialect_legitimacy=args.dialect_legitimacy,

        )
        system_content = next(
            (m["content"] for m in dummy_msgs if m["role"] == "system"), None
        )
        if system_content:
            backend.create_cache(system_content, args.model)
        else:
            print("Warning: could not extract system content for caching.")
        
        cache_end = time.time()
        print(f"Cache created in {cache_end - cache_start:.1f}s")
    # -------------------- COUNTERS --------------------
    usable_ctx_count    = 0
    used_two_turn_count = 0
    used_single_turn_count = 0
    dump_counter = 0  # Track number of prompts dumped

    start_time = time.time()
    print(f"\nStart: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # ==================== MAIN LOOP ====================
    for idx, sentence in enumerate(tqdm(eval_sentences, desc="Annotating")):
        if idx in existing_done_idxs:
            continue

        left = right = None
        usable = False

        if args.context:
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

        # Increment dump counter if we dumped this prompt
        if args.dump_prompt and (args.dump_first_n == 0 or dump_counter < args.dump_first_n):
            dump_counter += 1

        # API rate-limit courtesy sleep (local models don't need this)
        if args.backend in ("openai", "gemini") and args.rate_limit_delay > 0:
            time.sleep(args.rate_limit_delay)

        if arm_used == "two_turn":
            used_two_turn_count += 1
        elif arm_used == "single_turn":
            used_single_turn_count += 1

        if not raw:
            writemeta(idx, sentence, usable, arm_used, context_included,
                      "EMPTYRESPONSE", "", "")
            continue
        try:
            vals, rats, missing = parse_output(raw, CURRENT_FEATURES, output_format=args.output_format)
            parse_status  = "OK"
            missing_count = len(missing)
            missing_keys  = ",".join(missing)
            
            # Extract confidence scores (OpenAI only)
            confidences = extract_feature_confidences(backend, CURRENT_FEATURES, vals)

        except Exception as exc:
            print(f"Parse error at idx {idx}: {exc}")
            parse_status  = "PARSEFAIL"
            missing_count = ""
            missing_keys  = ""
            vals = {f: None for f in CURRENT_FEATURES}
            rats = {f: ""   for f in CURRENT_FEATURES}
            confidences = {}

        if isinstance(missing_count, int) and missing_count > 5:
            print(f"WARNING: {missing_count} missing features at idx {idx}")
        
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
        
        # Save confidence scores (OpenAI only)
        if confidences:
            with open(conf_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [idx, sentence] + [confidences.get(feat, "") for feat in CURRENT_FEATURES]
                )
    # -------------------- FINAL SUMMARY --------------------
    print_final_usage_summary(tracker) 

    end_time = time.time()
    print(f"End:     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")


if __name__ == "__main__":
    main()
