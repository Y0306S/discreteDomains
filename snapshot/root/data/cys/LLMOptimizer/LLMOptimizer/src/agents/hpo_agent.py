import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd
from openai import OpenAI


@dataclass
class HPOParameter:
    name: str
    value: Union[float, int, str]


@dataclass
class SingleConfigRecommendation:
    reasoning_trace: str
    warm_start_config: List[HPOParameter]
    raw_response: Optional[str] = None  # Full LLM output (only stored for thinking models or failures)


@dataclass
class HPORecommendation:
    recommendations: List[SingleConfigRecommendation]


@dataclass
class FailedRecommendation:
    error_reason: str
    raw_response: Optional[str] = None  # Full LLM output for debugging


class HPOAgent:
    def __init__(
        self,
        api_key,
        model: str = "gpt-5-mini",
        base_url: str = "https://api.openai-hk.com/v1",
        timeout: float = 180.0,
        num_ctx: int = 8192,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        from profiler import DataProfiler
        self.profiler = DataProfiler()
        self.current_profile = None

    def _is_local_model(self) -> bool:
        """Check if using Ollama, including through a tunnel/proxy URL."""
        base_url_str = str(self.base_url).lower().rstrip("/")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "").lower().rstrip("/")
        ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        return (
            "localhost" in base_url_str
            or "127.0.0.1" in base_url_str
            or (bool(ollama_base_url) and base_url_str == ollama_base_url)
            or str(self.api_key) == ollama_api_key
        )
    
    def _is_thinking_model(self) -> bool:
        """Check if the model is a thinking/reasoning model."""
        model_lower = self.model.lower()
        # Qwen models with thinking capability
        if 'qwen' in model_lower:
            return True
        # Claude thinking variants
        if 'claude' in model_lower and 'thinking' in model_lower:
            return True
        return False
    
    def _extract_content_from_response(self, message) -> str:
        """
        Extract content from response message, handling thinking/reasoning models.
        Some models (Qwen or Claude thinking variants) may return empty content but
        have reasoning_content or thinking blocks.
        """
        raw_json = message.content
        
        # If content is empty, check for reasoning_content (Qwen style)
        if not raw_json or raw_json.strip() == "":
            reasoning = getattr(message, 'reasoning_content', None)
            if reasoning:
                return reasoning
            
            # Check for thinking blocks (Claude style)
            thinking = getattr(message, 'thinking', None)
            if thinking:
                return thinking
            
            # Check for reasoning field (some API formats)
            reasoning = getattr(message, 'reasoning', None)
            if reasoning:
                return reasoning
        
        return raw_json or ""
    
    def _has_thinking_tags(self, content: str) -> bool:
        """Check if content contains thinking tags like <thinking>...</thinking> or <think>...</think>."""
        if not content:
            return False
        content_lower = content.lower()
        return any(tag in content_lower for tag in ['<thinking>', '</thinking>', '<think>', '</think>'])

    @staticmethod
    def _extract_json_payload(raw: str) -> dict:
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some local models may return valid JSON followed by extra text.
            # Recover by decoding the first JSON object in the string.
            start = text.find("{")
            if start == -1:
                raise
            decoder = json.JSONDecoder()
            payload, _ = decoder.raw_decode(text[start:])
            if not isinstance(payload, dict):
                raise ValueError("Recovered JSON payload is not an object.")
            return payload

    @staticmethod
    def _extract_recommendations_from_payload(payload: dict) -> list:
        """
        Extract individual recommendations from payload.
        Handles both:
        - Batch format: {"recommendations": [{...}, {...}]}
        - Single format: {"reasoning_trace": "...", "warm_start_config": [...]}
        Returns a list of individual recommendation dicts.
        """
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object.")
        
        # Check for batch wrapper format
        if "recommendations" in payload:
            recs = payload["recommendations"]
            if isinstance(recs, list):
                return recs
            raise ValueError("'recommendations' must be a list.")
        
        # Single format - wrap in list
        return [payload]

    @staticmethod
    def _to_single_recommendation(payload: dict) -> SingleConfigRecommendation:
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object.")
        trace = payload.get("reasoning_trace", "")
        cfg = payload.get("warm_start_config", [])
        if not str(trace).strip():
            raise ValueError("reasoning_trace is required and cannot be empty.")
        if not isinstance(cfg, list):
            raise ValueError("warm_start_config must be a list.")
        params = []
        for item in cfg:
            if not isinstance(item, dict):
                continue
            if "name" not in item or "value" not in item:
                continue
            params.append(HPOParameter(name=str(item["name"]), value=item["value"]))
        if not params:
            raise ValueError("warm_start_config must contain at least one parameter.")
        return SingleConfigRecommendation(reasoning_trace=str(trace), warm_start_config=params, raw_response=None)

    def load_hpobench(self, benchmark_obj):
        X_train, y_train, X_test, y_test = None, None, None, None

        if hasattr(benchmark_obj, "train_X"):
            X_train = benchmark_obj.train_X
            y_train = benchmark_obj.train_y
            X_test = getattr(benchmark_obj, "test_X", None)
            y_test = getattr(benchmark_obj, "test_y", None)
        elif hasattr(benchmark_obj, "get_data"):
            try:
                data = benchmark_obj.get_data()
                if len(data) == 4:
                    X_train, y_train, X_test, y_test = data
            except Exception:
                pass

        if X_train is None:
            raise ValueError("Could not extract data from Benchmark object.")

        def to_df(obj):
            if obj is None:
                return None
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj
            return pd.DataFrame(obj)

        X_train = to_df(X_train)
        X_test = to_df(X_test)
        y_train = to_df(y_train)
        y_test = to_df(y_test)

        if X_test is not None:
            X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
            y = pd.concat([y_train, y_test], axis=0, ignore_index=True)
        else:
            X, y = X_train, y_train

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        return X, y

    def analyze(self, benchmark_obj, intent="", task_id=None):
        X, y = self.load_hpobench(benchmark_obj)
        self.current_profile = self.profiler.profile(X, y, user_desc=intent, task_id=task_id)
        return self.current_profile

    def recommend_one(
        self,
        model_name: str = "unknown",
        config_space_info: str = "",
        previous_results: Optional[list] = None,
        cot_prompt: str = "",
    ):
        if not self.current_profile:
            raise ValueError("Run .analyze() first.")

        previous_results = previous_results or []
        meta = self.current_profile["meta"]
        class_distribution = ", ".join(
            [f"'{label}': {pct:.4f}%" for label, pct in meta.get("class_distribution", {}).items()]
        ) or "N/A"
        feature_breakdown = meta.get("feature_breakdown", {})

        prev_block = ""
        if previous_results:
            prev_lines = [
                "### PREVIOUS RESULTS",
                "The following configurations are already evaluated.",
                "Use these outcomes to guide your next recommendation.",
            ]
            for i, res in enumerate(previous_results, start=1):
                prev_lines.append(
                    f"Run {i}: loss={res.get('loss')}, accuracy={res.get('accuracy', 'N/A')}"
                )
                prev_lines.append(f"  config: {json.dumps(res.get('config', {}), default=str)}")
            prev_block = "\n".join(prev_lines)

        prompt = f"""
        You are an expert AutoML Architect specialising in hyperparameter optimisation (HPO).
        The model has already been selected: **{model_name.upper()}**.

        ### TASK & METRIC
        This is a **{meta['task_type']}** problem.
        The optimization metric is **'{meta['metric_name']}'** ({meta['metric_direction']}).

        ### DATASET PROFILE
        - **Intent:** {meta['description']}
        - **Dimensions:** {meta['rows']} samples, {meta['features']} total features
        - **Feature Types:** {feature_breakdown.get('numeric', 0)} numeric, {feature_breakdown.get('categorical_or_string', 0)} categorical/string, {feature_breakdown.get('boolean', 0)} boolean, {feature_breakdown.get('other', 0)} other
        - **Target Classes:** {meta.get('num_classes', 1)}
        - **Class Distribution:** {class_distribution}
        - **Missingness (sparsity):** {meta['sparsity'] * 100:.1f}%
        - **Task Type Source:** {meta.get('task_type_source', 'inferred')}

        ### HYPERPARAMETER SPACE
        {config_space_info}

        {prev_block}

        ### INSTRUCTIONS
        {cot_prompt}
        1. Recommend EXACTLY ONE warm-start configuration for **{model_name.upper()}**.
        2. If previous run results are available, adapt your proposal from them.
        3. Provide a detailed **reasoning_trace** with:
           a) dataset analysis,
           b) parameter recommendations,
           c) explicit connection between traits and choices.
        4. Use ONLY parameter names in the HYPERPARAMETER SPACE section.
        5. Never output placeholder names such as param_1, param_2, CONFIG_PARAMETERS.*, or descriptive aliases.

        ### STRICT OUTPUT FORMAT
        Return ONLY valid JSON in this exact shape. Replace every name with an exact parameter name copied from HYPERPARAMETER SPACE:
        {{
          "reasoning_trace": "Detailed reasoning here",
          "warm_start_config": [
            {{"name": "exact_parameter_name_from_hyperparameter_space", "value": 1}},
            {{"name": "another_exact_parameter_name_from_hyperparameter_space", "value": "choice"}}
          ]
        }}
        """

        _retry_waits = [30, 60, 120]
        
        # Determine model type for message formatting
        is_local = self._is_local_model()
        is_thinking = self._is_thinking_model()
        
        # Build messages based on model type
        # Thinking models may need special handling to avoid empty content
        messages = [
            {"role": "system", "content": "You are an expert AutoML Architect. You output only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        
        for attempt, wait in enumerate([0] + _retry_waits):
            if wait:
                time.sleep(wait)
            try:
                # Build API call parameters
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.8,
                    "timeout": 180.0,
                }
                
                # For thinking models (Claude thinking, etc.), top_p must be >= 0.95 or unset
                if is_thinking:
                    api_params["top_p"] = 1.0
                
                # Request JSON from both hosted and local OpenAI-compatible servers.
                api_params["response_format"] = {"type": "json_object"}
                
                # For local Ollama models, set num_ctx to control context size
                if is_local:
                    api_params["extra_body"] = {"num_ctx": self.num_ctx}
                
                response = self.client.chat.completions.create(**api_params)
                
                # Get raw content from response
                raw_content = response.choices[0].message.content or ""
                
                # Extract content, handling thinking/reasoning models
                raw_json = self._extract_content_from_response(response.choices[0].message)
                
                # Determine if we should store raw_response
                store_raw = self._has_thinking_tags(raw_content) or not raw_json
                
                if not raw_json:
                    reason = "Empty response from LLM (model may have run out of tokens during reasoning)"
                    print(f"  [WARN] {reason}")
                    parsed = FailedRecommendation(error_reason=reason, raw_response=raw_content if store_raw else None)
                else:
                    try:
                        parsed = self._to_single_recommendation(self._extract_json_payload(raw_json))
                        # Add raw_response if thinking tags detected
                        if store_raw:
                            parsed.raw_response = raw_content
                    except Exception as parse_exc:
                        reason = f"JSON parsing failed: {parse_exc}"
                        print(f"  [WARN] {reason}")
                        if raw_content:
                            preview = raw_content[:300].replace("\n", "\\n")
                            print(f"  [WARN] Raw response preview: {preview}")
                        parsed = FailedRecommendation(error_reason=reason, raw_response=raw_content if store_raw else None)

                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                return parsed, token_usage
            except Exception as e:
                if "429" in str(e) and attempt < len(_retry_waits):
                    continue
                raise

    def _request_recommendations(
        self,
        prompt: str,
        n: int = 1,
    ):
        """
        Internal method to request n recommendations from the LLM.
        Returns a list of SingleConfigRecommendation and token usage.
        """
        _retry_waits = [30, 60, 120]
        
        # Determine model type for message formatting
        is_local = self._is_local_model()
        is_thinking = self._is_thinking_model()
        
        # Build messages based on model type
        if is_thinking and is_local:
            combined_prompt = (
                "You are an expert AutoML Architect. "
                "You must output ONLY valid JSON, with no markdown formatting, no explanations outside the JSON, "
                "and no text before or after the JSON object. "
                "Start your response with '{' and end with '}'.\n\n"
                + prompt
            )
            messages = [{"role": "user", "content": combined_prompt}]
        else:
            messages = [
                {"role": "system", "content": "You are an expert AutoML Architect. You output only valid JSON."},
                {"role": "user", "content": prompt},
            ]
        
        for attempt, wait in enumerate([0] + _retry_waits):
            if wait:
                time.sleep(wait)
            try:
                # Build API call parameters
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.8,
                    "n": n,
                    "timeout": 180.0,
                }
                
                # For thinking models (Claude thinking, etc.), top_p must be >= 0.95 or unset
                if is_thinking:
                    api_params["top_p"] = 1.0
                
                # Request JSON from both hosted and local OpenAI-compatible servers.
                api_params["response_format"] = {"type": "json_object"}
                
                # For local Ollama models, set num_ctx to control context size
                if is_local:
                    api_params["extra_body"] = {"num_ctx": self.num_ctx}
                
                response = self.client.chat.completions.create(**api_params)
                
                all_recs = []
                for idx, choice in enumerate(response.choices, start=1):
                    # Get raw content from response
                    raw_content = choice.message.content or ""
                    
                    # Extract content, handling thinking/reasoning models
                    raw_json = self._extract_content_from_response(choice.message)
                    
                    # Determine if we should store raw_response:
                    # 1. If thinking tags are detected
                    # 2. If content is empty (failure case)
                    # 3. If JSON parsing fails
                    store_raw = self._has_thinking_tags(raw_content) or not raw_json
                    
                    if not raw_json:
                        reason = f"Empty response from LLM (model may have run out of tokens during reasoning)"
                        print(f"  [WARN] Candidate #{idx}: {reason}")
                        all_recs.append(FailedRecommendation(error_reason=reason, raw_response=raw_content if store_raw else None))
                        continue
                    
                    try:
                        payload = self._extract_json_payload(raw_json)
                        # Handle both batch wrapper format and single format
                        individual_payloads = self._extract_recommendations_from_payload(payload)
                        for individual_payload in individual_payloads:
                            try:
                                parsed = self._to_single_recommendation(individual_payload)
                                # Add raw_response if thinking tags detected
                                if store_raw:
                                    parsed.raw_response = raw_content
                                all_recs.append(parsed)
                            except Exception as parse_exc:
                                reason = f"JSON parsing failed: {parse_exc}"
                                print(f"  [WARN] Candidate #{idx}: {reason}")
                                if raw_content:
                                    preview = raw_content[:300].replace("\n", "\\n")
                                    print(f"  [WARN] Candidate #{idx} raw preview: {preview}")
                                all_recs.append(FailedRecommendation(error_reason=reason, raw_response=raw_content if store_raw else None))
                    except Exception as parse_exc:
                        reason = f"JSON parsing failed: {parse_exc}"
                        print(f"  [WARN] Candidate #{idx}: {reason}")
                        if raw_content:
                            preview = raw_content[:300].replace("\n", "\\n")
                            print(f"  [WARN] Candidate #{idx} raw preview: {preview}")
                        all_recs.append(FailedRecommendation(error_reason=reason, raw_response=raw_content if store_raw else None))
                
                # Truncate to requested n if we got more (batch wrapper can have multiple)
                recs = all_recs[:n] if len(all_recs) > n else all_recs
                
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                return recs, token_usage
            except Exception as e:
                if "429" in str(e) and attempt < len(_retry_waits):
                    continue
                raise

    def recommend_batch(
        self,
        model_name: str = "unknown",
        config_space_info: str = "",
        n: int = 3,
        previous_results: Optional[list] = None,
        cot_prompt: str = "",
    ):
        """
        Request n recommendations in a single batch call.
        Falls back to sequential calls if the API doesn't support n>1.
        """
        if not self.current_profile:
            raise ValueError("Run .analyze() first.")
        if n < 1:
            raise ValueError("n must be >= 1")

        previous_results = previous_results or []
        meta = self.current_profile["meta"]
        class_distribution = ", ".join(
            [f"'{label}': {pct:.4f}%" for label, pct in meta.get("class_distribution", {}).items()]
        ) or "N/A"
        feature_breakdown = meta.get("feature_breakdown", {})

        prev_block = ""
        if previous_results:
            prev_lines = [
                "### PREVIOUS RESULTS",
                "The following configurations are already evaluated.",
                "Use these outcomes to guide your next recommendation.",
            ]
            for i, res in enumerate(previous_results, start=1):
                prev_lines.append(
                    f"Run {i}: loss={res.get('loss')}, accuracy={res.get('accuracy', 'N/A')}"
                )
                prev_lines.append(f"  config: {json.dumps(res.get('config', {}), default=str)}")
            prev_block = "\n".join(prev_lines)

        prompt = f"""
        You are an expert AutoML Architect specialising in hyperparameter optimisation (HPO).
        The model has already been selected: **{model_name.upper()}**.

        ### TASK & METRIC
        This is a **{meta['task_type']}** problem.
        The optimization metric is **'{meta['metric_name']}'** ({meta['metric_direction']}).

        ### DATASET PROFILE
        - **Intent:** {meta['description']}
        - **Dimensions:** {meta['rows']} samples, {meta['features']} total features
        - **Feature Types:** {feature_breakdown.get('numeric', 0)} numeric, {feature_breakdown.get('categorical_or_string', 0)} categorical/string, {feature_breakdown.get('boolean', 0)} boolean, {feature_breakdown.get('other', 0)} other
        - **Target Classes:** {meta.get('num_classes', 1)}
        - **Class Distribution:** {class_distribution}
        - **Missingness (sparsity):** {meta['sparsity'] * 100:.1f}%
        - **Task Type Source:** {meta.get('task_type_source', 'inferred')}

        ### HYPERPARAMETER SPACE
        {config_space_info}

        {prev_block}

        ### INSTRUCTIONS
        {cot_prompt}
        1. Recommend EXACTLY ONE warm-start configuration for **{model_name.upper()}**.
        2. Provide a detailed **reasoning_trace** with:
           a) dataset analysis,
           b) parameter recommendations,
           c) explicit connection between traits and choices.
        3. Use ONLY parameter names in the HYPERPARAMETER SPACE section.
        4. Never output placeholder names such as param_1, param_2, CONFIG_PARAMETERS.*, or descriptive aliases.

        ### STRICT OUTPUT FORMAT
        Return ONLY valid JSON in this exact shape. Replace every name with an exact parameter name copied from HYPERPARAMETER SPACE:
        {{
          "reasoning_trace": "Detailed reasoning here",
          "warm_start_config": [
            {{"name": "exact_parameter_name_from_hyperparameter_space", "value": 1}},
            {{"name": "another_exact_parameter_name_from_hyperparameter_space", "value": "choice"}}
          ]
        }}
        """

        # Always use sequential n=1 calls to get individual reasoning for each config
        recs = []
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for i in range(n):
            try:
                call_recs, tok = self._request_recommendations(prompt=prompt, n=1)
                recs.append(call_recs[0])
                for k in total_tokens:
                    total_tokens[k] += tok[k]
            except Exception as e:
                # If a single call fails, add a FailedRecommendation
                reason = f"API call failed for recommendation {i+1}: {str(e)}"
                print(f"  [WARN] {reason}")
                recs.append(FailedRecommendation(error_reason=reason, raw_response=None))
        
        return recs, total_tokens, prompt

    def recommend(
        self,
        model_name: str = "unknown",
        config_space_info: str = "",
        fidelity_space_info: str = "",
        cot_prompt: str = "",
    ):
        """
        Backward-compatible helper that returns three one-shot recommendations.
        Uses batch call for efficiency.
        """
        _ = fidelity_space_info
        recs, total_tokens, _ = self.recommend_batch(
            model_name=model_name,
            config_space_info=config_space_info,
            n=3,
            previous_results=[],
            cot_prompt=cot_prompt,
        )
        return HPORecommendation(recommendations=recs), total_tokens
