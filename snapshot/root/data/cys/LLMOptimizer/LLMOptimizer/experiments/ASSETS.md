# Assets and deployment

The source bundle does not replace the artifact store. Before a full run, supply:

| Campaign | Required external assets |
| --- | --- |
| Query evaluation | CEB SQL workloads/schema, populated PostgreSQL IMDB instances and credentials, task split and measured source corpus, original query VAE, adapted VAE/FSBO checkpoints, Qwen base/BOLT/ORPO checkpoints, retrieval cards and source GP bank, inference/VAE service configs |
| Peptide gated FSBO | Source800 payload snapshot and integrity manifest, split/recipe, initial100 sequences and measured scores for each target, adapted offline VAE/FSBO checkpoints, APEX oracle weights, source retention validation data, execution allocation |
| Peptide hybrid | Qwen base/BOLT/ORPO/RL checkpoints, retrieval GP bank and cards, frozen FSBO checkpoint, the peptide oracle/VAE assets and initialization pool |
| Training | Source800 measured corpus, accepted training splits, training examples/pairs, tokenizer/base checkpoint, compatible CUDA/LLM training environment and LLaMA-Factory installation for the SFT entry points |

The main query campaign uses BOLT30 then ORPO+FSBO batch10 through 80 total rounds (800 observations); RL is disabled in that evaluation. The gated peptide test uses 100 initialization observations plus up to 4,000 acquisition proposals, batch50 through 80 rounds, patience20, and at most two retraining attempts. Cached proposals and fresh oracle evaluations are recorded separately.

## Preparing a deployment

1. Make a fresh runtime campaign directory outside this source tree. Preserve `code/`, `batch_code/`, and `training/` layouts used by the chosen campaign; do not merge modules named `common`, `method`, or `run` across campaigns.
2. Populate the external assets and update the historical paths in the chosen Python snapshot. `deployment_paths.json` lists the configuration variables; `ENVIRONMENT.example` shows the main named asset roots. Query and peptide native `lolbo` packages must run in separate Python processes or namespaces.
3. Render the selected JSON template into the runtime directory with `render_config.py`. For example, after setting its required variables:

   ```bash
   python experiments/render_config.py \
     experiments/peptide_fsbo/configs/config.template.json \
     /your/new/campaign/config.json
   ```

   This command creates a new file and refuses to overwrite an existing identity. It does not generate assets, service configurations, validation evidence, or updated integrity pins.
4. Install dependencies for the chosen runtime. `environment_versions.cpu_checks.json` records the environment used for the bundled CPU checks; it is not a tested CUDA inference lockfile. The two `requirements.txt` files are dependency inventories, not universal pins. Optional RGPE uses its separate upstream environment.
5. Rerun unit tests and the selected campaign's live/saved-checkpoint validators in the destination. Generate fresh deployment code pins and resource/service configuration. Controllers enforce their original preflight checks; copied historical validation JSON is provenance only.
6. Run a small evaluation on a training task, confirm oracle/DB measurements and checkpoint/resume behavior, then launch the controller. In particular, never point a new deployment at the live production output directories.

The original per-campaign READMEs retain historical launch details and paths. They document the source campaigns rather than an automatically configured new host. The live experiments were not restarted or modified by packaging.
