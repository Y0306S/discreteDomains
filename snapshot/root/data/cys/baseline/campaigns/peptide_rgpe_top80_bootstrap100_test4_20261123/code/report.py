"""Verify the completed pilot and summarize timing separately from optimization quality."""
import json
from pathlib import Path
import statistics
from source_screen import ROOT, atomic, digest

OLD = ROOT.parent/'peptide_rgpe_screen200_test200_20261119'
TASKS = ['pep_104', 'pep_109', 'pep_111', 'pep_132']


def read(path):
    return json.loads(path.read_text())


def model_seconds(fit, screen, sample):
    return fit['seconds']+screen['seconds']+sample['merge_seconds']+sample['sample_seconds']


def main():
    completion = read(ROOT/'evidence/pilot_processes.json')
    assert completion['passed']
    results = []
    for task in TASKS:
        history = read(ROOT/'rgpe/runs'/task/'history.json')
        baseline = read(OLD/'rgpe/runs'/task/'history.json')
        initial = history['observations'][:100]
        assert [(r['sequence'], r['score']) for r in initial] == [
            (r['sequence'], r['score']) for r in baseline['observations'][:100]]
        assert history['rounds'] == 2 and history['events'] == 2 and history['proposals'] == 100
        assert len(history['observations']) == 200
        selection = read(ROOT/'prefilter'/task/'selection.json')
        selected = selection['selected_tasks']
        assert len(set(selected)) == 80
        rounds = []
        last_elapsed = 0.
        for number in (1, 2):
            telemetry = read(ROOT/'telemetry'/task/('round%d.json'%number))
            fit, screen, sample = [telemetry[k] for k in ('fit', 'screen', 'sample')]
            assert fit['target_observations'] == 100+50*(number-1)
            assert fit['source_models'] == 80 and fit['source_tasks'] == selected
            assert fit['source_selection_sha256'] == digest(ROOT/'prefilter'/task/'selection.json')
            assert fit['sampling_mode'] == 'bootstrap' and fit['num_posterior_samples'] == 100
            assert len(fit['weights']) == 81 and abs(sum(fit['weights'])-1) < 1e-10
            assert screen['candidates'] == 1500 and screen['shortlisted'] == 200
            assert sample['candidates'] == 200 and sample['samples'] == 50
            rows = [r for r in history['observations'] if r['round'] == number]
            assert len(rows) == 50
            seen = history['observations'][:100+50*number]
            elapsed = max(r['elapsed'] for r in rows)
            rounds.append(dict(round=number,
                model_recommendation_seconds=model_seconds(fit, screen, sample),
                acquisition_seconds=telemetry['acquisition_seconds'],
                accounted_event_seconds=elapsed-last_elapsed,
                target_fit_seconds=fit['target_fit_seconds'], bootstrap_seconds=fit['bootstrap_seconds'],
                merge_seconds=sample['merge_seconds'], target_weight=fit['target_weight'],
                nonzero_weights=fit['nonzero_weights'], source_cache_hits=sample['source_cache_hits'],
                source_cache_misses=sample['source_cache_misses'], proposals=50,
                fresh=sum(not r['cached'] for r in rows), feasible=sum(r['feasible'] for r in rows),
                fresh_feasible=sum(r['feasible'] and not r['cached'] for r in rows),
                best_feasible_mic=min(r['mic'] for r in seen if r['feasible']),
                best_batch_mic=min(r['mic'] for r in rows)))
            last_elapsed = elapsed
        baseline_seconds = model_seconds(
            read(OLD/'telemetry'/task/'fit_seed1.json')['info'],
            read(OLD/'telemetry'/task/'screen_seed1.json')['info'],
            read(OLD/'telemetry'/task/'sample_seed1.json')['info'])
        losses = sorted(s['loss'] for s in selection['sources'])
        results.append(dict(task=task, initialization_observations=100,
            initialization_best_feasible_mic=min(r['mic'] for r in initial),
            best_feasible_mic=history['best_feasible_mic'], fresh=history['fresh_evaluations'],
            rounds=rounds, fixed_selection_sha256=digest(ROOT/'prefilter'/task/'selection.json'),
            selection_loss_min=losses[0], selection_loss_cutoff=losses[79],
            selection_loss_max=losses[-1], ties_at_cutoff=losses.count(losses[79]),
            weights_changed=read(ROOT/'telemetry'/task/'round1.json')['fit']['weights'] !=
                            read(ROOT/'telemetry'/task/'round2.json')['fit']['weights'],
            baseline_first_round_model_seconds=baseline_seconds,
            first_round_model_speedup=baseline_seconds/rounds[0]['model_recommendation_seconds']))
    report = dict(pilot_complete=True, tasks=results,
        one_time_shared_source_screen_seconds=read(ROOT/'prefilter/complete.json')['total_seconds'],
        parallel_pilot_process_wall_seconds=completion['elapsed'],
        sampling_mode='bootstrap', bootstrap_resamples=100, sources_per_target=80,
        membership='fixed_after_initialization', ensemble_weights='recomputed_each_round',
        initialization=100, acquisition_rounds=2, batch_size=50,
        comparison_limits='Prior Nov19 runs have only one completed round. Hardware/cache/load and algorithm differ; timing ratios are observed, not controlled speedup estimates.',
        weight_observation='All four first rounds assign target weight 1. Round2 only pep_104 retains sources: pep_166=0.10, pep_712=0.05, target=0.85; all others target=1. Posterior merge mostly skips historical models, so the large speedup cannot be attributed solely to top80 filtering.',
        timing_definitions=dict(model_recommendation_seconds='Target GP fit + bootstrap weights + EI screen + posterior merge + joint draw; excludes constraint GP fit and peptide evaluation.',
            accounted_event_seconds='Existing runner acquisition/decoding/oracle accounting; excludes canonical reencoding/persistence.',
            parallel_pilot_process_wall_seconds='Four coordinators run concurrently; includes initialization and all evaluations, excludes one-time source screen.'),
        optimization_budget_preserved=4000, persisted_history_status='running: stopped intentionally after 2 events, full optimization budget not exhausted')
    atomic(ROOT/'summary.json', report)
    lines = ['# Peptide RGPE fixed top80 / bootstrap100 pilot', '',
        '100 initialization observations → source mean-only ordered-pair ranking → fixed top80 → bootstrap100 weights recomputed every round.', '',
        'Four tasks, two rounds each, 50 proposals per round. Original frozen VAE, APEX, 1500→200 target-EI shortlist and 4000-step dilution horizon preserved.', '',
        'One-time shared source screen: %.2f min; four-task parallel pilot wall time: %.2f min.'%(
            report['one_time_shared_source_screen_seconds']/60, completion['elapsed']/60), '',
        '| Task | Init best feasible MIC | Round1 MIC | Round2 MIC | Acquisition round1 / round2 (s) | Fresh / fresh feasible |',
        '|---|---:|---:|---:|---:|---:|']
    for result in results:
        a,b=result['rounds']
        lines.append('| %s | %.4f | %.4f | %.4f | %.2f / %.2f | %d / %d |'%(
            result['task'], result['initialization_best_feasible_mic'], a['best_feasible_mic'],
            b['best_feasible_mic'], a['acquisition_seconds'], b['acquisition_seconds'],
            result['fresh'], a['fresh_feasible']+b['fresh_feasible']))
    lines += ['', 'Lower MIC is better. Feasibility uses the original peptide similarity constraint.', '',
        'Each prefilter/<task>/selection.json contains the fixed 80 source IDs and all 800 ranking losses. telemetry/<task>/round{1,2}.json records all 81 weights and per-stage timing.', '',
        report['weight_observation'], '',
        'Acquisition timing includes candidate generation, worker initialization (first call), GP fit, bootstrap weighting, EI shortlist, merged-posterior sampling and constraint sampling; excludes the preceding constraint GP fit and subsequent peptide evaluation. All four tasks yielded zero fresh feasible peptides; pep_132 repeated one already-cached feasible sequence.', '',
        'summary.json gives the earlier experiment’s first-round timing for reference. It is not a controlled benchmark: hardware, cache/load, source count and ranking mode differ. Two rounds do not establish long-run optimization quality.', '',
        'Validation: upstream ranking parity including ties; full-GP vs compact mean-only bootstrap weight parity on three real GPs; identical initial sequences/scores and reused canonical coordinates; all four tasks committed exactly two rounds. Full-horizon history status remains running because this finite pilot stops at 2 events.', '']
    (ROOT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
