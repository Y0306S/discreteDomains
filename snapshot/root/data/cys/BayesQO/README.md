# BayesQO: Learned Offline Query Planning via Bayesian Optimization

## How to Run Optimization

1. (Optional) Set up a weights and biases (wandb) account so you can automatically track optimization progress and results

   https://wandb.ai/home

   Your account will be set up with a wandb entity name (like a short username). The default scripts set the wandb entity to my entity name (nmaus-penn). Chance this to your entity name by passing in $YOUR_WANDB_ENTITY with the --wandb_entity argument.

2. Clone repository using git lfs to also clone all large files

3. Install python dependencies using your favorite pyproject.toml-compatible tool, e.g., `uv sync -U`

4. Optimization needs to execute against a PostgreSQL database pre-loaded with the target dataset. A docker container for testing can be built from `imdb_postgres/`. Supply the connection details using environment variables:

   - `DB_HOST=172.17.0.2`
   - `DB_USER=imdb`
   - `DB_PASSWORD=imdb`.

5. Run the following script to start an optimization run for a particular workload specified by the --workload_name argument.

```Bash
cd bayes_lqo/lolbo_scripts
```

If want to track with weights and biases, this command is fine. Otherwise, set the env var `WANDB_MODE="offline"` or `WANDB_MODE="disabled"`.

```Bash
python info_transformer_vae_optimization.py --workload_name JOB_16B --allow_cross_joins False --init_w_bao True --wandb_entity haydnj --so_future True --force_past_vae False - run_lolbo - done
```

Workload names are i.e. JOB_6F, JOB_9D, JOB_19D, JOB_7C, JOB_8C, JOB_17F, JOB_16B, etc. A list of JOB workloads can be found at [tasks/job_tasks.txt](tasks/job_tasks.txt).

## Paper

If you use this code, please cite our [SIGMOD 2025 paper](https://dl.acm.org/doi/10.1145/3725316).

    @article{10.1145/3725316,
        author = {Tao, Jeffrey and Maus, Natalie and Jones, Haydn and Zeng, Yimeng and Gardner, Jacob R. and Marcus, Ryan},
        title = {Learned Offline Query Planning via Bayesian Optimization},
        year = {2025},
        issue_date = {June 2025},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {3},
        number = {3},
        url = {https://doi.org/10.1145/3725316},
        doi = {10.1145/3725316},
        journal = {Proc. ACM Manag. Data},
        month = jun,
        articleno = {179},
        numpages = {29}
    }
