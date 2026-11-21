import time
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from openai import OpenAI

@dataclass
class NASParameter:
    name: str
    value: Union[float, int, str]

@dataclass
class SingleRecommendation:
    reasoning_trace: str
    warm_start_config: List[NASParameter]
    raw_response: Optional[str] = None  # Full LLM output (only stored for thinking models or failures)

@dataclass
class FailedRecommendation:
    error_reason: str
    raw_response: Optional[str] = None  # Full LLM output for debugging

@dataclass
class NASRecommendation:
    recommendations: List[SingleRecommendation]

class TabularNASAgent:
    def __init__(
        self,
        api_key,
        model: str = "gpt-5-mini",
        base_url: str = "https://api.openai-hk.com/v1",
        timeout: float = 180.0,
        num_ctx: int = 8192,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.num_ctx = num_ctx
        self.current_meta = None
        self._nas101_allowed_ops = {
            "conv3x3-bn-relu",
            "conv1x1-bn-relu",
            "maxpool3x3",
        }

    @staticmethod
    def _extract_json_payload(raw: str) -> dict:
        text = (raw or "").strip()
        
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                # Find the closing ```
                for i, line in enumerate(lines[1:], 1):
                    if line.strip().startswith("```"):
                        text = "\n".join(lines[1:i]).strip()
                        break
                else:
                    # No closing found, try without first line
                    text = "\n".join(lines[1:]).strip()
        
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find and extract the first JSON object
        decoder = json.JSONDecoder()
        start = text.find("{")
        
        # Try multiple starting positions in case there's text before JSON
        while start != -1:
            try:
                payload, end_idx = decoder.raw_decode(text, start)
                if isinstance(payload, dict):
                    return payload
                # If it's not a dict, continue searching
                start = text.find("{", start + 1)
            except json.JSONDecodeError:
                # Try next position
                start = text.find("{", start + 1)
        
        # Try finding JSON array as fallback
        start = text.find("[")
        while start != -1:
            try:
                payload, end_idx = decoder.raw_decode(text, start)
                if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], dict):
                    return payload[0]
                start = text.find("[", start + 1)
            except json.JSONDecodeError:
                start = text.find("[", start + 1)
        
        raise ValueError(f"No valid JSON object found in response. Raw text: {text[:200]}...")

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
    def _to_single_recommendation(payload: dict) -> SingleRecommendation:
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object.")
        trace = payload.get("reasoning_trace", "")
        cfg = payload.get("warm_start_config", [])
        if not isinstance(cfg, list):
            raise ValueError("warm_start_config must be a list.")
        params = []
        for item in cfg:
            if not isinstance(item, dict):
                continue
            if "name" not in item or "value" not in item:
                continue
            params.append(NASParameter(name=str(item["name"]), value=item["value"]))
        return SingleRecommendation(reasoning_trace=str(trace), warm_start_config=params, raw_response=None)

    def _to_nasbench101_recommendation(self, payload: dict) -> SingleRecommendation:
        if not isinstance(payload, dict):
            raise ValueError("Invalid NAS-Bench-101 output: response must be a JSON object.")

        reasoning = str(payload.get("reasoning", payload.get("reasoning_trace", "")))
        architecture = payload.get("architecture", {})
        if not isinstance(architecture, dict):
            raise ValueError("Invalid NAS-Bench-101 output: architecture must be an object.")

        adjacency = architecture.get("module_adjacency")
        operations = architecture.get("module_operations")

        if not isinstance(adjacency, list) or len(adjacency) != 7:
            raise ValueError("Invalid NAS-Bench-101 output: module_adjacency must be a 7x7 array.")
        for row in adjacency:
            if not isinstance(row, list) or len(row) != 7:
                raise ValueError("Invalid NAS-Bench-101 output: module_adjacency must be a 7x7 array.")

        normalized_adj = [[0 for _ in range(7)] for _ in range(7)]
        edge_count = 0
        for i in range(7):
            for j in range(7):
                try:
                    val = int(adjacency[i][j])
                except Exception as exc:
                    raise ValueError("Invalid NAS-Bench-101 output: module_adjacency entries must be 0 or 1.") from exc
                if val not in (0, 1):
                    raise ValueError("Invalid NAS-Bench-101 output: module_adjacency entries must be 0 or 1.")
                if i >= j and val != 0:
                    raise ValueError("Invalid NAS-Bench-101 output: module_adjacency must be upper triangular (strict DAG).")
                normalized_adj[i][j] = val
                edge_count += val

        if edge_count > 9:
            raise ValueError("Invalid NAS-Bench-101 output: total edges cannot exceed 9.")

        if not isinstance(operations, list) or len(operations) != 7:
            raise ValueError("Invalid NAS-Bench-101 output: module_operations must contain 7 entries.")
        operations = [str(x) for x in operations]
        if operations[0] != "input" or operations[-1] != "output":
            raise ValueError("Invalid NAS-Bench-101 output: module_operations must start with 'input' and end with 'output'.")
        for op in operations[1:6]:
            if op not in self._nas101_allowed_ops:
                raise ValueError(f"Invalid NAS-Bench-101 output: unsupported operation '{op}'.")

        # Reachability check: there must be at least one path from input(0) to output(6).
        visited = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for nxt in range(7):
                if normalized_adj[node][nxt] == 1 and nxt not in visited:
                    visited.add(nxt)
                    frontier.append(nxt)
        if 6 not in visited:
            raise ValueError("Invalid NAS-Bench-101 output: no path from input to output.")

        metadata = payload.get("metadata")
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError("Invalid NAS-Bench-101 output: metadata must be an object.")
            if "total_edges" in metadata and int(metadata["total_edges"]) != edge_count:
                raise ValueError(
                    "Invalid NAS-Bench-101 output: metadata.total_edges does not match module_adjacency."
                )
            if "unique_nodes" in metadata and int(metadata["unique_nodes"]) != 7:
                raise ValueError(
                    "Invalid NAS-Bench-101 output: metadata.unique_nodes must be 7."
                )

        params: List[NASParameter] = []
        edge_idx = 0
        for i in range(7):
            for j in range(i + 1, 7):
                params.append(NASParameter(name=f"edge_{edge_idx}", value=int(normalized_adj[i][j])))
                edge_idx += 1
        for node_idx, op in enumerate(operations[1:6]):
            params.append(NASParameter(name=f"op_node_{node_idx}", value=op))

        return SingleRecommendation(reasoning_trace=reasoning, warm_start_config=params, raw_response=None)

    @staticmethod
    def _is_nasbench101(benchmark_name: str) -> bool:
        return str(benchmark_name) == "NASCifar10A"

    def _build_prompt(
        self,
        benchmark_name: str,
        config_space_info: str,
        previous_results: Optional[list],
        task_description: str,
        targets_and_parameters: str,
        dataset_profile: str,
    ) -> Tuple[str, bool]:
        is_nas101 = self._is_nasbench101(benchmark_name)
        previous_results = previous_results or []
        prev_block = ""
        if previous_results:
            lines = [
                "### Previous Results",
                "These runs are already evaluated. Improve your next proposal using these outcomes.",
            ]
            for i, res in enumerate(previous_results, start=1):
                lines.append(f"Run {i}: loss={res.get('loss')}")
                lines.append(f"  config: {json.dumps(res.get('config', {}), default=str)}")
            prev_block = "\n".join(lines)

        if is_nas101:
            strict_output = """
### STRICT OUTPUT FORMAT
You must respond with ONLY valid JSON matching this exact structure.
No markdown, no comments, and no extra text before or after JSON.

Before finalizing your answer, internally verify every rule below:
1. architecture.module_adjacency is a 7x7 matrix.
2. Every module_adjacency entry is either 0 or 1.
3. The matrix is strictly upper triangular: for all i >= j, adjacency[i][j] == 0.
4. Total number of edges (sum of all 1s) is <= 9.
5. There is at least one directed path from node 0 (input) to node 6 (output).
6. architecture.module_operations has exactly 7 entries.
7. module_operations[0] must be "input" and module_operations[6] must be "output".
8. module_operations[1..5] must each be one of:
   - "conv3x3-bn-relu"
   - "conv1x1-bn-relu"
   - "maxpool3x3"
9. metadata.total_edges must equal the number of 1s in module_adjacency.
10. metadata.unique_nodes must be 7.
If any rule fails, fix the architecture and re-check internally before responding.

Return this exact JSON shape:
{
  "reasoning": "Your detailed reasoning here.",
  "architecture": {
    "module_adjacency": [
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 0]
    ],
    "module_operations": [
      "input",
      "conv3x3-bn-relu",
      "conv1x1-bn-relu",
      "maxpool3x3",
      "conv3x3-bn-relu",
      "conv1x1-bn-relu",
      "output"
    ]
  },
  "metadata": {
    "total_edges": 6,
    "unique_nodes": 7
  }
}
"""
        else:
            strict_output = """
### STRICT OUTPUT FORMAT
You must respond with ONLY valid JSON matching this exact structure.
Every name must be copied exactly from the hyperparameter space. Do not use placeholders such as param_1, param_2, CONFIG_PARAMETERS.*, or descriptive aliases.
{
  "reasoning_trace": "Your detailed reasoning here.",
  "warm_start_config": [
    {"name": "exact_parameter_name_from_hyperparameter_space", "value": 0.5},
    {"name": "another_exact_parameter_name_from_hyperparameter_space", "value": "categorical_choice"}
  ]
}
"""

        prompt = f"""
You are an expert AutoML Architect specialising in Neural Architecture Search.

### Part 1: Task Description
{task_description}

### Part 2: Targets & Parameters
{targets_and_parameters}

### Part 3: Dataset Profile
{dataset_profile}

### Part 4: Hyperparameter Space
{config_space_info}

{prev_block}

### Instructions
1. Recommend EXACTLY ONE warm-start architecture/configuration.
2. If previous results are provided, adapt based on what worked and what failed.
3. Use ONLY parameter names listed in the hyperparameter space.
4. For categorical parameters, choose strictly from provided choices.
5. Explain your reasoning in detail and connect parameter choices to dataset/task traits.

{strict_output}
"""
        return prompt, is_nas101

    def _is_local_model(self) -> bool:
        """Check if using Ollama, including through a tunnel/proxy URL."""
        base_url = getattr(self.client, '_base_url', '') or self.base_url
        base_url_str = str(base_url).lower().rstrip("/")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "").lower().rstrip("/")
        ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        return (
            "localhost" in base_url_str
            or "127.0.0.1" in base_url_str
            or (bool(ollama_base_url) and base_url_str == ollama_base_url)
            or str(self.api_key) == ollama_api_key
        )
    
    def _is_qwen_model(self) -> bool:
        """Check if the model is a Qwen model (which uses ChatML format and has reasoning_content)."""
        model_lower = self.model.lower()
        return 'qwen' in model_lower
    
    def _is_thinking_model(self) -> bool:
        """Check if the model is a thinking/reasoning model that may have reasoning_content."""
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

    def _request_recommendations(
        self,
        prompt: str,
        is_nas101: bool,
        n: int = 1,
    ):
        _retry_waits = [30, 60, 120]
        
        # Determine if we can use JSON mode (OpenAI API supports it, local models may not)
        is_local = self._is_local_model()
        is_qwen = self._is_qwen_model()
        is_thinking = self._is_thinking_model()
        
        # Build messages based on model type
        # Thinking models (Qwen or Claude thinking variants) may output reasoning in reasoning_content field
        # and actual response in content. To avoid empty content due to token limits,
        # we use a single user message for thinking models instead of system+user.
        if (is_qwen or is_thinking) and is_local:
            # For local thinking models: combine system and user into single user message
            # This avoids triggering extensive reasoning mode that may exhaust tokens
            combined_prompt = (
                "You are an expert AutoML Architect. "
                "You must output ONLY valid JSON, with no markdown formatting, no explanations outside the JSON, "
                "and no text before or after the JSON object. "
                "Start your response with '{' and end with '}'.\n\n"
                + prompt
            )
            messages = [{"role": "user", "content": combined_prompt}]
        elif is_local:
            # For other local models
            system_content = (
                "You are an expert AutoML Architect. "
                "You must output ONLY valid JSON, with no markdown formatting, no explanations outside the JSON, "
                "and no text before or after the JSON object. "
                "Start your response with '{' and end with '}'."
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
        else:
            # For OpenAI API models
            system_content = "You are an expert AutoML Architect. You output only valid JSON."
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
        
        for attempt, wait in enumerate([0] + _retry_waits):
            if wait:
                print(f"    [Rate Limit] Waiting for {wait} seconds before retry {attempt}...")
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
                
                # Only use response_format for non-local models (OpenAI API)
                if not is_local:
                    api_params["response_format"] = {"type": "json_object"}
                
                # For local Ollama models, set num_ctx to control context size
                if is_local:
                    api_params["extra_body"] = {"num_ctx": self.num_ctx}
                
                response = self.client.chat.completions.create(**api_params)
                # Handle case where response might be a string (error message)
                if isinstance(response, str):
                    raise RuntimeError(f"API returned string instead of object: {response[:500]}")
                
                # Debug: Log how many choices were returned
                print(f"    [Debug] Requested n={n}, received {len(response.choices)} choice(s)")
                
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
                    
                    # If still empty after extraction, it's a failed response
                    if not raw_json:
                        reason = f"Empty response from LLM (model may have run out of tokens during reasoning)"
                        print(f"  [WARN] Skipping invalid LLM output — Candidate #{idx}: {reason}")
                        all_recs.append(FailedRecommendation(error_reason=reason, raw_response=raw_content if store_raw else None))
                        continue
                    
                    try:
                        payload = self._extract_json_payload(raw_json)
                        # Handle both batch wrapper format and single format
                        if is_nas101:
                            # NAS-101 has its own format, don't use batch wrapper extraction
                            parsed = self._to_nasbench101_recommendation(payload)
                            # Add raw_response if thinking tags detected
                            if store_raw:
                                parsed.raw_response = raw_content
                            all_recs.append(parsed)
                        else:
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
                                    print(f"  [WARN] Skipping invalid LLM output — Candidate #{idx}: {reason}")
                                    all_recs.append(FailedRecommendation(error_reason=reason, raw_response=raw_content if store_raw else None))
                    except Exception as parse_exc:
                        reason = f"JSON parsing failed: {parse_exc}"
                        print(f"  [WARN] Skipping invalid LLM output — Candidate #{idx}: {reason}")
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

    def analyze(self, benchmark_obj):
        """
        Extracts metadata directly from the tabular benchmarks
        """
        try:
            self.current_meta = benchmark_obj.get_meta_information()
        except Exception as e:
            print(f"    Could not extract meta information: {e}")
            self.current_meta = {"info": "metadata not available for this benchmark."}
        return self.current_meta

    def recommend_one(
        self,
        benchmark_name: str,
        config_space_info: str,
        previous_results: Optional[list] = None,
        task_description: str = "",
        targets_and_parameters: str = "",
        dataset_profile: str = "",
    ):
        prompt, is_nas101 = self._build_prompt(
            benchmark_name=benchmark_name,
            config_space_info=config_space_info,
            previous_results=previous_results,
            task_description=task_description,
            targets_and_parameters=targets_and_parameters,
            dataset_profile=dataset_profile,
        )
        recs, token_usage = self._request_recommendations(prompt=prompt, is_nas101=is_nas101, n=1)
        return recs[0], token_usage

    def recommend_batch(
        self,
        benchmark_name: str,
        config_space_info: str,
        n: int = 3,
        previous_results: Optional[list] = None,
        task_description: str = "",
        targets_and_parameters: str = "",
        dataset_profile: str = "",
    ):
        """
        Get n recommendations by making n separate n=1 calls.
        This ensures each recommendation has its own reasoning trace and raw response.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        prompt, is_nas101 = self._build_prompt(
            benchmark_name=benchmark_name,
            config_space_info=config_space_info,
            previous_results=previous_results,
            task_description=task_description,
            targets_and_parameters=targets_and_parameters,
            dataset_profile=dataset_profile,
        )

        # Always use sequential n=1 calls to get individual reasoning for each config
        recs = []
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for i in range(n):
            try:
                print(f"  [LLM] Requesting recommendation {i + 1}/{n}...")
                call_recs, tok = self._request_recommendations(prompt=prompt, is_nas101=is_nas101, n=1)
                recs.append(call_recs[0])
                for k in total_tokens:
                    total_tokens[k] += tok[k]
            except Exception as e:
                # If a single call fails, add a FailedRecommendation
                reason = f"API call failed for recommendation {i+1}: {str(e)}"
                print(f"  [WARN] {reason}")
                recs.append(FailedRecommendation(error_reason=reason, raw_response=None))
        
        return recs, total_tokens, prompt

    def recommend(self, benchmark_name: str, config_space_info: str):
        """
        Backward-compatible method that returns 3 recommendations.
        """
        recs, total_tokens, _ = self.recommend_batch(
            benchmark_name=benchmark_name,
            config_space_info=config_space_info,
            n=3,
            previous_results=[],
        )
        return NASRecommendation(recommendations=recs), total_tokens
