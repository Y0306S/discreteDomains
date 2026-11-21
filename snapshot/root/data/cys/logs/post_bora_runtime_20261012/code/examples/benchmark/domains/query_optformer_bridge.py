"""SeqIO registration for approved CEB windows, separate from ML training data."""

import json
from pathlib import Path

import numpy as np

from query_transfer_common import digest, load_corpus, window_indices


def study_from_window(record):
    from vizier import pyvizier as vz
    problem = vz.ProblemStatement()
    problem.metadata['name'] = record['task_id']
    problem.metadata['algorithm'] = record['behavior_policy']
    problem.metric_information.append(vz.MetricInformation(name='score', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    for i in range(64):
        problem.search_space.root.add_float_param('z_%02d' % i, 0.0, 1.0)
    trials = []
    for i, (x, y) in enumerate(zip(record['X'], record['y_maximize'])):
        trial = vz.Trial(id=i + 1, parameters={'z_%02d' % j: value for j, value in enumerate(x)})
        trial.complete(vz.Measurement(metrics={'score': y}))
        trials.append(trial)
    return vz.ProblemAndTrials(problem=problem, trials=trials)


def load_windows(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'manifest.json').read_text())
    corpus = Path(meta['corpus'])
    manifest, sources = load_corpus(corpus)
    if (meta['corpus_sha256'] != digest(corpus / 'manifest.json')
            or meta['windows_sha256'] != digest(directory / 'windows.jsonl')
            or meta['source_tasks'] != manifest['source_tasks'] or meta['max_observations'] != 32):
        raise ValueError('Query window provenance mismatch')
    expected = [(task, indices) for task, source in sources.items()
                for indices in window_indices(len(source['X']), 32)]
    with (directory / 'windows.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    if len(records) != len(expected) or len(records) != meta['windows']:
        raise ValueError('Query window count mismatch')
    for record, (task, indices) in zip(records, expected):
        source = sources[task]
        if (record['task_id'] != task or not record['ordered'] or record['shuffled']
                or record['source_rows'] != [source['source_rows'][i] for i in indices]
                or not np.array_equal(record['X'], source['X'][indices])
                or not np.array_equal(record['y_maximize'], source['y_maximize'][indices])):
            raise ValueError('Query window changed its source observations or chronology')
    return meta, manifest, records


def register_task(manifest):
    import seqio
    from optformer.data import tasks
    from mindware.components.optimizers.transfer_hpo.optformer_bridge import make_converter, checked_conversion

    directory = Path(manifest['windows'])
    if (manifest['method'] != 'optformer_query'
            or manifest['windows_manifest_sha256'] != digest(directory / 'manifest.json')
            or manifest['vocabulary_sha256'] != digest(Path(manifest['vocabulary']))):
        raise ValueError('Query OptFormer training manifest mismatch')
    _, corpus, records = load_windows(directory)
    if corpus['vae_sha256'] != manifest['vae_sha256']:
        raise ValueError('Query OptFormer VAE mismatch')
    vocabulary = tasks.get_vocabulary(manifest['vocabulary'], expected_vocab_size=33100)

    def dataset_fn(split, shuffle_files=False, seed=None):
        del shuffle_files
        if split not in ('train', 'train_monitor'):
            raise ValueError('No held-out query data is registered with the trainer')
        if seed is not None:
            np.random.seed(seed)
        converter = make_converter(training=split == 'train')

        class CheckedConverter:
            def study_to_texts(self, study):
                return checked_conversion(converter, study, vocabulary, manifest['lengths'])[0]

        def generator():
            for record in records:
                yield study_from_window(record)

        return tasks.vizier_dataset_from_generator(generator, CheckedConverter(), suppress_error=False)

    tasks.add_tasks('query_train800', vocabulary=vocabulary, masked_types=['separator'],
                    dataset_fn=dataset_fn, splits=('train', 'train_monitor'))
    task = seqio.TaskRegistry.get('query_train800').replace(shuffle_buffer_size=len(records))
    seqio.TaskRegistry.remove('query_train800')
    seqio.TaskRegistry.add_provider('query_train800', task)
    return task
