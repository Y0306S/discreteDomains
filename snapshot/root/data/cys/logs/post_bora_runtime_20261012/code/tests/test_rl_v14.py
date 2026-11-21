"""v14 RL collection/training rules: group membership, resume guard, KL, loader checks."""

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import contextlib
import inspect
import re
import tempfile

import torch

BENCH = Path(__file__).resolve().parents[1] / "examples" / "benchmark"
sys.path.insert(0, str(BENCH))

import run_p5_rl_bo_prefix_b as collector  # noqa: E402
import train_p5_reinforce as trainer  # noqa: E402

TEST_MANIFEST = Path("/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_48h_20260925/tasks_132.json")
TRAIN_MANIFEST = Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl")
ALLOWLIST = Path("/root/reports/rl_plan_20261005/train32_tasks.json")


@contextlib.contextmanager
def raises(exc, match):
    try:
        yield
    except exc as e:
        assert re.search(match, str(e)), f"{e!r} does not match {match!r}"
        return
    raise AssertionError(f"{exc.__name__} not raised")


def rec(i, status="SUCCESS", delta=-0.2, reward=None):
    return {"sample_i": i, "evaluation_status": status,
            "excluded_from_training": status in ("TIMEOUT", "EVAL_FAILED", "GENERATION_ERROR"),
            "reward": {"delta_best": delta, "reward": delta if reward is None else reward}}


def choose(records):
    return collector.choose_train_members_v14(records, n_base=4, rng=random.Random(0))


def test_improver_in_first_four_keeps_first_four():
    rs = [rec(0, delta=0.3), rec(1), rec(2, delta=-0.5), rec(3, delta=-0.1)]
    assert choose(rs)[0] == [0, 1, 2, 3]


def test_late_improver_replaces_a_random_member():
    rs = [rec(i, delta=-0.1 * (i + 1)) for i in range(4)] + [rec(4, delta=-0.9), rec(5, delta=0.4)]
    train, dropped, injected, status = choose(rs)
    assert injected == 5 and dropped in range(4) and 5 in train and len(train) == 4 and status == "ok"


def test_timeouts_are_replaced_by_later_samples_and_never_train():
    rs = [rec(0, delta=0.2), rec(1, "TIMEOUT"), rec(2, delta=-0.3), rec(3, delta=-0.1), rec(4, delta=-0.4)]
    train, *_ = choose(rs)
    assert train == [0, 2, 3, 4]


def test_duplicate_or_parse_failure_never_counts_as_improver():
    rs = [rec(0, "DUPLICATE", delta=5.0, reward=-0.5), rec(1, "PARSE_FAILURE", delta=5.0, reward=-1.0),
          rec(2, delta=-0.2), rec(3, delta=-0.3), rec(4, delta=0.6)]
    train, dropped, injected, _ = choose(rs)
    assert injected == 4


def test_all_timeouts_drop_group():
    assert choose([rec(i, "TIMEOUT") for i in range(10)])[3] == "all_excluded"


def test_identical_rewards_carry_no_signal():
    rs = [rec(i, "DUPLICATE", reward=-0.5) for i in range(4)]
    train, *_, status = choose(rs)
    assert train == [] and status == "no_reward_signal"


def test_torn_group_blocks_resume(tmp_path):
    path = tmp_path / "rollouts_x.jsonl"
    rows = [{"prefix_k": 3, "scenario": "rerank", "n_sampled": 4} for _ in range(3)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with raises(RuntimeError, match="Incomplete groups"):
        collector._existing_job_counts(path)


class Toy(torch.nn.Module):
    def __init__(self, vocab=11, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = torch.nn.Parameter(torch.randn(vocab, vocab, generator=g))

    def forward(self, input_ids, logits_to_keep=None):
        logits = self.table[input_ids]
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


def test_full_vocab_kl_matches_definition_and_has_gradient():
    policy, ref = Toy(seed=1), Toy(seed=2)
    ids = torch.tensor([[1, 4, 2, 7, 3, 9]])
    labels = ids.clone()
    labels[0, :3] = -100
    logp, kl, ref_logp = trainer.answer_logprob_and_kl(policy, ref, ids, labels, chunk=2)
    lp = torch.log_softmax(policy.table[ids[0, 2:5]], -1)
    lr = torch.log_softmax(ref.table[ids[0, 2:5]], -1)
    targets = ids[0, 3:]
    assert torch.allclose(logp, lp.gather(-1, targets[:, None]).mean())
    assert torch.allclose(kl, (lr.exp() * (lr - lp)).sum(-1).mean())
    kl.backward()
    assert policy.table.grad.abs().sum() > 0


def test_kl_is_zero_for_identical_models():
    model = Toy(seed=3)
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = ids.clone()
    labels[0, :2] = -100
    _, kl, _ = trainer.answer_logprob_and_kl(model, Toy(seed=3), ids, labels)
    assert abs(float(kl)) < 1e-6


def v14_row(**over):
    task = json.loads(ALLOWLIST.read_text())[0]
    label = "classification" if task["task_type"] == "cls" else "regression"
    row = {
        "schema": "rl_rollout.v14", "task_type": task["task_type"], "dataset": task["dataset"],
        "algorithm": task["algorithm"], "group_id": "g1", "sample_i": 0, "trial": "k3_s0",
        "messages": [{"role": "system", "content": "First write a short rationale, then write one JSON object"},
                     {"role": "user", "content": f"Dataset profile:\n- task_type: {label}\n\nSearch space:\n- a:b: integer, range=[0, 5]\n\nRespond as Rationale: ... then JSON: {{...}}"}],
        "completion": 'Rationale: x\n\nJSON:\n{"config": {"a:b": 1}}', "response_format": "rationale_json",
        "chat_template_contract": trainer.CHAT_TEMPLATE_VERSION, "evaluation_status": "SUCCESS",
        "train_member": True, "excluded_from_training": False, "reward": {"reward": 0.1}, "llm_max_tokens": 768,
    }
    row.update(over)
    return row


def write_buffer(tmp_path, rows):
    (tmp_path / "rollouts_t.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def load(tmp_path):
    return trainer.load_records_v14(tmp_path, allowlist=ALLOWLIST, train_manifest=TRAIN_MANIFEST,
                                    test_manifest=TEST_MANIFEST, expected_response_format="rationale_json")


def test_loader_rejects_mislabelled_profile(tmp_path):
    bad = v14_row()
    bad["messages"][1]["content"] = bad["messages"][1]["content"].replace(
        "classification", "regression") if bad["task_type"] == "cls" else bad["messages"][1]["content"].replace(
        "regression", "classification")
    write_buffer(tmp_path, [bad])
    with raises(SystemExit, match="Dataset profile"):
        load(tmp_path)


def test_loader_rejects_test_task(tmp_path):
    test_task = json.loads(TEST_MANIFEST.read_text())[0]
    write_buffer(tmp_path, [v14_row(task_type=test_task["task_type"], dataset=test_task["dataset"],
                                    algorithm=test_task["algorithm"])])
    with raises(SystemExit, match="split"):
        load(tmp_path)


def test_loader_rejects_duplicate_rollouts(tmp_path):
    write_buffer(tmp_path, [v14_row(), v14_row()])
    with raises(SystemExit, match="duplicate rollout"):
        load(tmp_path)


def test_loader_drops_groups_without_signal(tmp_path):
    write_buffer(tmp_path, [v14_row(sample_i=0), v14_row(sample_i=1)])
    records, stats = load(tmp_path)
    assert records == [] and stats["groups_dropped_no_signal"] == 1


def test_loader_keeps_parse_failures_as_negatives(tmp_path):
    rows = [v14_row(sample_i=0, reward={"reward": 0.3}),
            v14_row(sample_i=1, evaluation_status="PARSE_FAILURE", completion="garbled", reward={"reward": -1.0})]
    records, stats = load(tmp_path_rows(tmp_path, rows))
    assert sorted(r["status"] for r in records) == ["PARSE_FAILURE", "SUCCESS"]


def tmp_path_rows(tmp_path, rows):
    write_buffer(tmp_path, rows)
    return tmp_path


def test_loader_never_trains_timeouts(tmp_path):
    rows = [v14_row(sample_i=0, reward={"reward": 0.3}),
            v14_row(sample_i=1, evaluation_status="TIMEOUT", excluded_from_training=True, reward={"reward": 0.0}),
            v14_row(sample_i=2, reward={"reward": -0.2})]
    records, _ = load(tmp_path_rows(tmp_path, rows))
    assert all(r["status"] != "TIMEOUT" for r in records) and len(records) == 2


def test_truncated_completion_gets_no_eos():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B", local_files_only=True)
    messages = [{"role": "user", "content": "x"}]
    text = "Rationale: " + "word " * 50
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    full, _ = trainer.build_batch_ids(tok, messages, text, torch.device("cpu"), 10 ** 6, generation_limit=n + 10)
    cut, _ = trainer.build_batch_ids(tok, messages, text, torch.device("cpu"), 10 ** 6, generation_limit=n)
    assert full[0, -1].item() == tok.eos_token_id
    assert cut[0, -1].item() != tok.eos_token_id and cut.shape[1] == full.shape[1] - 1


def test_kl_with_reference_on_another_device():
    if torch.cuda.device_count() < 2:
        return
    policy, ref = Toy(seed=1).to("cuda:0"), Toy(seed=2).to("cuda:1")
    ids = torch.tensor([[1, 4, 2, 7, 3, 9]], device="cuda:0")
    labels = ids.clone()
    labels[0, :3] = -100
    logp, kl, _ = trainer.answer_logprob_and_kl(policy, ref, ids, labels)
    cpu_logp, cpu_kl, _ = trainer.answer_logprob_and_kl(policy.cpu(), ref.cpu(), ids.cpu(), labels.cpu())
    assert torch.allclose(logp.cpu(), cpu_logp, atol=1e-5) and torch.allclose(kl.cpu(), cpu_kl, atol=1e-5)


def test_space_violations_are_detected_from_the_prompt():
    prompt = ("Search space:\n- a:n: integer, range=[50, 500]\n- a:x: float, range=[0.5, 1.0]\n"
              "- a:c: categorical, choices=['True', 'False']\n")
    def row(cfg):
        return {"messages": [{"role": "user", "content": prompt}],
                "completion": "Rationale: r\n\nJSON:\n" + json.dumps({"config": cfg})}
    ok = {"a:n": 100, "a:x": 0.7, "a:c": "True"}
    assert trainer.invalid_proposal_reason(row(ok)) is None
    assert "outside" in trainer.invalid_proposal_reason(row(dict(ok, **{"a:n": 545})))
    assert "outside" in trainer.invalid_proposal_reason(row(dict(ok, **{"a:x": 0.41})))
    assert "not numeric" in trainer.invalid_proposal_reason(row(dict(ok, **{"a:x": "True"})))
    assert "not in" in trainer.invalid_proposal_reason(row(dict(ok, **{"a:c": "maybe"})))
    assert "missing" in trainer.invalid_proposal_reason(row({"a:n": 100, "a:c": "True"}))
    assert "integer" in trainer.invalid_proposal_reason(row(dict(ok, **{"a:n": 100.5})))


def test_v15_schedule_is_dense_early_and_drops_late_checkpoints():
    import p5plus_rl_collect_schedule as sched
    ks = [k for k, _, _ in sched.mix_jobs_v15()]
    assert ks == [3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
    assert {sc for k, sc, _ in sched.mix_jobs_v15() if k <= 7} == {"rerank"}
    args = SimpleNamespace(collect_schedule="rs_mix_v15", rs_version="v3", only_prefixes=None)
    assert [k for k, _, _ in collector.collect_schedule_steps(args)] == ks


def test_std_floor_keeps_near_ties_weak():
    near_tie = torch.tensor([0.2010, 0.2015, 0.2020, 0.2030])
    assert trainer._standardize(near_tie).abs().max() > 1.0          # v14 behaviour: full-size advantages
    assert trainer._standardize(near_tie, 0.05).abs().max() < 0.03   # v15: a weak signal
    spread = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    assert torch.allclose(trainer._standardize(spread, 0.05), trainer._standardize(spread))


def test_success_reward_floor_clips_only_successes(tmp_path):
    rows = [v14_row(sample_i=0, reward={"reward": -4.5}),
            v14_row(sample_i=1, reward={"reward": 0.3}),
            v14_row(sample_i=2, evaluation_status="PARSE_FAILURE", completion="garbled", reward={"reward": -1.0})]
    write_buffer(tmp_path, rows)
    records, stats = trainer.load_records_v14(tmp_path, allowlist=ALLOWLIST, train_manifest=TRAIN_MANIFEST,
                                              test_manifest=TEST_MANIFEST, expected_response_format="rationale_json",
                                              success_reward_floor=-1.0)
    rewards = sorted(r["reward"] for r in records)
    assert rewards == [-1.0, -1.0, 0.3] and stats["success_rewards_clipped"] == 1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failures += 1
                print("FAIL", name, repr(e))
    print(f"{failures} failed")
    raise SystemExit(1 if failures else 0)
