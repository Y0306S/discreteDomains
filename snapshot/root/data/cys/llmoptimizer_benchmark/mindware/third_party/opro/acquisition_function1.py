import os
import time
import openai
import asyncio
import numpy as np
import pandas as pd
from aiohttp import ClientSession
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from mindware.third_party.opro.rate_limiter import RateLimiter
from typing import Any, Dict, List, Tuple

from .utils import (
    _serialize_row,
    _prepare_configurations_acquisition,  # 仍保留可用，但这里自建带 Label 的版本
    _constraints_text,
    _dataset_summary_block,               # 若需要，也可回退用它的文本
    _convert_to_dict,
    _filter_and_validate,
)

openai.api_type = os.environ.get("OPENAI_API_TYPE", "")
openai.api_version = os.environ.get("OPENAI_API_VERSION", "")
openai.api_base = os.environ.get("OPENAI_API_BASE", "")
openai.api_key = os.environ.get("OPENAI_API_KEY", "")


class LLM_ACQ:
    def __init__(
        self,
        task_context,
        n_candidates,
        n_templates,
        lower_is_better,
        jitter=False,
        rate_limiter=None,
        warping_transformer=None,
        chat_engine=None,
        prompt_setting=None,
        shuffle_features=False,
        # history / few-shot 控制
        history_budget: int = 30,
        top_k: int = 15,
        worst_k: int = 3,
        recent_k: int = 8,
        per_algo_cap: int = 10,
        interleave_order: tuple = ("best", "recent", "worst"),
    ):
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = max(1, int(n_candidates / max(1, n_templates)))
        self.lower_is_better = lower_is_better
        self.apply_jitter = jitter
        self.rate_limiter = rate_limiter or RateLimiter(max_tokens=800000000, time_frame=300)
        self.warping_transformer = warping_transformer
        self.apply_warping = warping_transformer is not None
        self.chat_engine = chat_engine
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

        # schema
        self.constraints: Dict[str, Dict[str, Any]] = task_context.get("hyperparameter_constraints", {})
        self.algo_choices: List[str] = task_context.get("algorithm_choices", [])
        self.algo2params: Dict[str, List[str]] = task_context.get("algorithm_param_names", {})

        # few-shot 选择策略（由本类读取这些参数并实现）
        self.history_budget = history_budget
        self.top_k = top_k
        self.worst_k = worst_k
        self.recent_k = recent_k
        self.per_algo_cap = per_algo_cap
        self.interleave_order = interleave_order

        # 记录统计信息（可选）
        self.observed_best = None
        self.observed_worst = None
        self.alpha = 0.2

    # =============== Prompt 相关构造 ===============

    def _build_system_prompt(self) -> str:
        """
        更明确地告诉 LLM：你在做 CASH（算法选择 + 超参搜索）以优化指定指标。
        """
        metric = self.task_context.get("metric", "score")
        direction = "minimize" if self.lower_is_better else "maximize"
        algo_choices_txt = (
            f" Choose one algorithm from {self.algo_choices} and set ONLY its parameters."
            if self.algo_choices else
            " Choose a valid algorithm and set ONLY its parameters."
        )
        return (
            "You are a model selection and hyperparameter optimization assistant. "
            f"Your job is to {direction} the evaluation metric '{metric}' by proposing a valid algorithm "
            f"and its hyperparameters. {algo_choices_txt} "
            "Always follow the search-space constraints and output exactly in the requested format."
        )

    def _build_dataset_summary_sentences(self) -> str:
        """
        用完整自然语言描述数据与任务。若 task_context 缺少信息，则尽量容错。
        """
        prof = self.task_context.get("dataset_profile", {}) or {}
        metric = self.task_context.get("metric", "score")

        # 任务/指标
        task_type = prof.get("task_type", "unknown task")
        parts = [f"This is a {task_type} problem. The optimization metric is '{metric}'."]
        # 规模/特征
        n_samples = prof.get("n_samples")
        n_features = prof.get("n_features")
        n_num = prof.get("n_numeric")
        n_cat = prof.get("n_categorical")
        if n_samples is not None and n_features is not None and n_num is not None and n_cat is not None:
            parts.append(
                f"The dataset contains {n_samples} samples and {n_features} features "
                f"({n_num} numeric and {n_cat} categorical)."
            )
        elif n_samples is not None and n_features is not None:
            parts.append(f"The dataset contains {n_samples} samples and {n_features} features.")

        # 缺失情况
        mr = prof.get("missing_rate", 0.0)
        if isinstance(mr, (int, float)) and mr > 0:
            parts.append(f"There is a non-zero missing rate of {mr:.4f}.")

        # 分类标签分布
        if prof.get("task_type") == "classification":
            cd = prof.get("class_distribution", {})
            dist = cd.get("distribution", {})
            if dist:
                # “Class A: 0.845, Class B: 0.155”风格
                dist_str = ", ".join([f"{str(k)}: {float(v):.3f}" for k, v in dist.items()])
                parts.append(f"Class distribution is: {dist_str}.")

        # 回归目标统计
        if prof.get("task_type") == "regression":
            ts = prof.get("target_stats", {})
            stats_frag = []
            for k in ["mean", "std", "min", "median", "max"]:
                if k in ts:
                    stats_frag.append(f"{k}={float(ts[k]):.4f}")
            if stats_frag:
                parts.append("Target statistics: " + ", ".join(stats_frag) + ".")

        return " ".join(parts)

    def _select_labeled_examples(self, observed_configs: pd.DataFrame, observed_fvals: pd.DataFrame):
        """
        选择 top-k / recent-k / worst-k，并返回带 Label 的 few-shot 样例：
        [{'Label': 'BEST', 'Q': '## ... ##', 'A': '0.863000'}, ...]
        同时做 per-algo 配额与总预算控制。
        """
        if observed_configs is None or observed_fvals is None or len(observed_configs) == 0:
            return []

        # 先算 score 列并排序
        df = observed_configs.copy()
        df["_score"] = observed_fvals.values.astype(float).ravel()
        ascending = True if self.lower_is_better else False
        df_sorted = df.sort_values("_score", ascending=ascending)

        # 分桶索引
        best_idx = list(df_sorted.index[: self.top_k])
        worst_idx = list(df_sorted.index[-self.worst_k:]) if self.worst_k > 0 else []
        recent_idx = list(df.index[-self.recent_k:]) if self.recent_k > 0 else []

        # 合并顺序
        buckets = {"best": best_idx, "recent": recent_idx, "worst": worst_idx}
        merged = []
        for key in self.interleave_order:
            for i in buckets.get(key, []):
                if i not in merged:
                    merged.append(i)

        # per-algo cap
        algo_col = "algorithm"
        algo_counts: Dict[str, int] = {}
        selected = []
        for i in merged:
            algo = df.loc[i, algo_col] if algo_col in df.columns else None
            if algo is None:
                # 尝试从参数名前缀推断
                for c in df.columns:
                    if ":" in c and not pd.isna(df.loc[i, c]):
                        algo = c.split(":", 1)[0]
                        break
            if algo is None:
                selected.append(i)
                continue
            cnt = algo_counts.get(algo, 0)
            if cnt < self.per_algo_cap:
                selected.append(i)
                algo_counts[algo] = cnt + 1

        # 总预算
        if len(selected) > self.history_budget:
            selected = selected[: self.history_budget]

        # 生成 few-shot examples，并打 Label
        examples = []
        best_set, recent_set, worst_set = set(best_idx), set(recent_idx), set(worst_idx)
        for i in selected:
            row = observed_configs.loc[i]
            A = float(observed_fvals.loc[i].values[0])
            if i in best_set:
                label = "BEST"
            elif i in recent_set:
                label = "RECENT"
            elif i in worst_set:
                label = "WORST"
            else:
                label = "EXAMPLE"
            examples.append({
                "Label": label,
                "Q": _serialize_row(self, row),
                "A": f"{A:.6f}",
            })
        return examples

    def _gen_prompt_tempates_acquisitions(
    self, observed_configs, observed_fvals, desired_fval, n_prompts=1, use_context="full_context"):
        """
        Prompt 顺序：
        1) 你在做什么/目标；
        2) 数据与任务的自然语言概述；
        3) 可选算法与搜索空间（constraints）；
        4) 历史样例（带 BEST/RECENT/WORST 标签 + 明确说明这些是“历史得到的性能-配置对”）；
        5) 当前目标性能 + 严格输出格式约束。
        """
        all_prompt_templates, all_query_templates = [], []

        for i in range(n_prompts):
            # few-shot 带标签
            few_shot_examples = self._select_labeled_examples(observed_configs, observed_fvals)

            # 每条 example 的模板
            example_template = (
                "{Label}\n"
                "Performance: {A}\n"
                "Hyperparameter configuration: {Q}"
            )
            example_prompt = PromptTemplate(
                input_variables=["Q", "A", "Label"],
                template=example_template,
            )

            # 1) 系统目标
            system_goal = self._build_system_prompt()

            # 2) 数据集自然语言概述（句子）
            data_summary = self._build_dataset_summary_sentences() if self.task_context.get(
                "add_dataset_summary_to_prompt", True
            ) else ""

            # 3) 搜索空间
            constraints_text = _constraints_text(self)

            # 4) 历史样例引导 + 标签解释（关键新增 history_intro）
            history_intro = (
                "The following are historical examples collected on this same dataset and task. "
                "Each block shows a past trial’s evaluation Performance and the corresponding model hyperparameter configuration."
            )
            label_guidance = (
                "Examples are labeled as follows:\n"
                "- BEST: top-performing configurations; explore around these regions.\n"
                "- RECENT: most recent trials; they indicate the current search trend.\n"
                "- WORST: poor configurations; generally avoid these directions.\n"
            )

            # 5) 目标与输出格式
            metric = self.task_context.get("metric", "score")
            direction = "lower is better" if self.lower_is_better else "higher is better"

            prefix_parts = [
                f"{system_goal}\n",
                (f"Dataset & Task:\n{data_summary}\n" if data_summary else ""),
                "Search space constraints:\n" + constraints_text + "\n",
                history_intro + "\n",
                label_guidance,
                f"Target: propose one configuration expected to achieve {desired_fval:.6f} {metric} "
                f"({direction}).\n"
                "Do not use boundary values. Use the highest precision allowed by the ranges.\n"
                "Do not copy any example verbatim; propose an improved configuration.\n"
                "Your response must ONLY contain the predicted configuration in the exact format:\n"
                "## algorithm: <name>, <algo>:<param>: <value>, <algo>:<param>: <value>, ... ##",
            ]
            prefix = "\n".join([p for p in prefix_parts if p])

            # suffix：把“要生成的目标性能 A”放在最后，随后是“Hyperparameter configuration:”
            suffix = "\nPerformance: {A}\nHyperparameter configuration:"

            few_shot_prompt = FewShotPromptTemplate(
                examples=few_shot_examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=["A"],
                example_separator="\n\n",
            )

            all_prompt_templates.append(few_shot_prompt)
            all_query_templates.append([{"A": f"{desired_fval:.6f}"}])

        return all_prompt_templates, all_query_templates

    # =============== LLM 调用 ===============
    async def _async_generate(self, user_message: str):
        message = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_message},
        ]
        MAX_RETRIES = 3
        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)
            resp = None
            for retry in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    self.rate_limiter.add_request(request_text=user_message, current_time=start_time)
                    resp = await openai.ChatCompletion.acreate(
                        model=self.chat_engine,
                        messages=message,
                        temperature=0.8,
                        max_tokens=500,
                        top_p=0.95,
                        n=self.n_gens,
                        request_timeout=30,
                    )
                    self.rate_limiter.add_request(
                        request_token_count=resp["usage"]["total_tokens"], current_time=start_time
                    )
                    break
                except Exception as e:
                    print(f"[AF] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...", e)
        await openai.aiosession.get().close()

        if resp is None:
            return None
        tot_tokens = resp["usage"]["total_tokens"]
        tot_cost = 0.0015 * (resp["usage"]["prompt_tokens"] / 1000) + 0.002 * (
            resp["usage"]["completion_tokens"] / 1000
        )
        return resp, tot_cost, tot_tokens

    async def _async_generate_concurrently(self, prompt_templates, query_templates):
        filled_prompts = [pt.format(A=qt[0]["A"]) for pt, qt in zip(prompt_templates, query_templates)]

        out_dir = self.task_context.get("output_dir")
        iter_idx = self.task_context.get("current_iter_idx", None)
        dump = self.task_context.get("dump_llm_templates", False)

        try:
            if dump and out_dir:
                prompts_dir = os.path.join(out_dir, "prompts")
                responses_dir = os.path.join(out_dir, "responses")
                os.makedirs(prompts_dir, exist_ok=True)
                os.makedirs(responses_dir, exist_ok=True)
                for j, msg in enumerate(filled_prompts):
                    fname = f"iter{iter_idx:04d}_tpl{j:02d}.txt" if isinstance(iter_idx, int) else f"tpl{j:02d}.txt"
                    with open(os.path.join(prompts_dir, fname), "w", encoding="utf-8") as f:
                        f.write(msg)
                    print(f"[LLM_ACQ] Saved prompt: {fname}, length: {len(msg)}")

            llm_responses = await asyncio.gather(*[self._async_generate(msg) for msg in filled_prompts])

            if dump and out_dir:
                responses_dir = os.path.join(out_dir, "responses")
                for idx, response in enumerate(llm_responses):
                    if response is None:
                        continue
                    for resp_idx, choice in enumerate(response[0]["choices"]):
                        filename = (
                            f"iter{iter_idx:04d}_tpl{idx:02d}_resp{resp_idx:02d}.txt"
                            if isinstance(iter_idx, int)
                            else f"tpl{idx:02d}_resp{resp_idx:02d}.txt"
                        )
                        resp_text = choice.get("message", {}).get("content", "")
                        with open(os.path.join(responses_dir, filename), "w", encoding="utf-8") as f:
                            f.write(resp_text)
                        print(f"[LLM_ACQ] Saved response: {filename}, length: {len(resp_text)}")

            return [r for r in llm_responses if r is not None]
        except Exception as e:
            print("[LLM_ACQ] prompt dump failed:", e)
            return []

    # =============== 主流程 ===============
    def get_candidate_points(
        self, observed_configs, observed_fvals, use_context="full_context", alpha=0.2, iter_idx=None
    ):
        assert -1 <= alpha <= 1
        if alpha == 0:
            alpha = -1e-3
        self.alpha = alpha

        if iter_idx is not None:
            self.task_context["current_iter_idx"] = int(iter_idx)

        # 设定目标性能（desired_fval）
        vals = observed_fvals.values.astype(float).ravel()
        rng = float(np.abs(np.max(vals) - np.min(vals)))
        if rng == 0:
            rng = 0.1 * float(np.abs(np.max(vals)))
        alpha_candidates = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, 1e-1]

        if self.lower_is_better:
            self.observed_best, self.observed_worst = float(np.min(vals)), float(np.max(vals))
            desired = self.observed_best - alpha * rng
            while desired <= 1e-6:
                for a in alpha_candidates:
                    if a < alpha:
                        alpha = a
                        desired = self.observed_best - alpha * rng
                        break
        else:
            self.observed_best, self.observed_worst = float(np.max(vals)), float(np.min(vals))
            desired = self.observed_best + alpha * rng
            while desired >= 0.9999:
                for a in alpha_candidates:
                    if a < alpha:
                        alpha = a
                        desired = self.observed_best + alpha * rng
                        break
        desired_fval = desired

        # 可选：warping
        warped_cfgs = self.warping_transformer.warp(observed_configs) if self.apply_warping else observed_configs

        # 构建 few-shot prompt（带 BEST / RECENT / WORST 标签 + 自然语言数据集描述 + 任务目标）
        prompt_templates, query_templates = self._gen_prompt_tempates_acquisitions(
            warped_cfgs, observed_fvals, desired_fval, n_prompts=self.n_templates, use_context=use_context
        )

        number_candidate_points = 0
        filtered_candidate_points = pd.DataFrame()
        retry, tot_cost, tot_tokens = 0, 0.0, 0
        t0_llm = time.time()

        while number_candidate_points < self.n_candidates:
            llm_responses = asyncio.run(self._async_generate_concurrently(prompt_templates, query_templates))
            if not llm_responses:
                break

            candidate_points: List[Dict[str, Any]] = []
            for resp, c_cost, c_tokens in llm_responses:
                for ch in resp.get("choices", []):
                    content = ch.get("message", {}).get("content", "") or ""
                    try:
                        first = content.find("##")
                        last = content.rfind("##")
                        if first != -1 and last != -1 and last > first:
                            segment = content[first + 2 : last].strip()
                        else:
                            segment = content.split("##")[1].strip()
                    except Exception:
                        continue
                    cand = _convert_to_dict(segment)
                    if cand:
                        candidate_points.append(cand)
                tot_cost += c_cost
                tot_tokens += c_tokens

            proposed = _filter_and_validate(self, warped_cfgs.to_dict(orient="records"), candidate_points)

            filtered_candidate_points = (
                pd.concat([filtered_candidate_points, proposed], ignore_index=True).drop_duplicates()
            )
            if filtered_candidate_points.shape[0] > self.n_candidates:
                filtered_candidate_points = filtered_candidate_points.head(self.n_candidates)

            number_candidate_points = filtered_candidate_points.shape[0]
            print(f"Attempt: {retry}, proposed: {len(candidate_points)}, accepted: {number_candidate_points}")
            retry += 1
            if retry > 3 and number_candidate_points == 0:
                raise Exception("LLM failed to generate candidate points")

        if self.apply_warping and filtered_candidate_points is not None and len(filtered_candidate_points) > 0:
            filtered_candidate_points = self.warping_transformer.unwarp(filtered_candidate_points)

        time_taken = time.time() - t0_llm
        return filtered_candidate_points, tot_cost, time_taken