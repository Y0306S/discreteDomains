Completed ML evaluation detail archive

The archive contains only evaluations/ and decisions/ for tasks whose saved state
was complete in ml_orpo_fsbo120_20261109 (120 ORPO and 117 ORPO+GRPO tasks).
Session checkpoint.pkl, progress.json, state, outputs, histories, and unfinished
tasks remain at their original paths.

Before removal, GNU tar compared every archived entry with its original.
The shared copy was also checked with SHA-256 against the local archive.
See cleanup_report.json and deleted_paths.jsonl for the actual removed paths.

Restore the detail directories when needed:

tar -xzf /root/data/cys/baseline/archives/completed_ml_evaluation_details_20261111.tar.gz -C /root/data/cys/baseline/campaigns/ml_orpo_fsbo120_20261109

KuaiRand was separately deleted at the user's explicit request and is not included
in this archive. The deletion record is /root/kuairand_deletion_20261111.json.
