# Validation on 21 November 2026

The following checks ran against the migration copy on the original server:

- All 31,229 captured source/configuration/workload file SHA256 hashes matched
  the inventory; ten required domain/native/baseline entry points were present.
- Nine Python environment metadata exports succeeded, including the Python 3.7
  RGPE environment. PostgreSQL 16.15 Conda package specifications were exported
  separately.
- Seventeen peptide CPU tests passed: target/source rollback behavior, exact
  parameter/buffer restoration, RNG/probe isolation, numerical failure handling,
  and the 20-round/two-attempt scheduling policy.
- Thirteen query CPU tests passed: separate retrieval encoder/cache, FSBO batch
  behavior, batch selection, novelty and measurement accounting.
- Sixty-one Python migration/current-campaign files parsed successfully, and
  all migration shell scripts passed `bash -n`.
- A full source-layout restore into a fresh `/tmp` directory succeeded without
  conflicts. The restore helper also supports preview and refuses overwrites.
- A fixture SQLite journal was backed up with its measured rows preserved and
  `quick_check` passing; conflicting restore files were correctly rejected.
- A local rsync fixture preserved the artifact directory layout and excluded
  source files already captured by Git. All 25 primary configured absolute
  input/model paths existed on the capture host.
- Gitleaks 8.30.1 reported zero findings after removing embedded API credentials
  from eight copied shared-helper files. `.gitleaks.toml` permits SHA256 file
  identities and the literal generated-event path format used by journals.

The domain tests used the existing server's installed environments and native
dependencies. They made no oracle calls and launched no training or evaluation
campaign. These results do not establish that a clean machine has all model
weights, database data, system libraries or sufficient hardware.

No environment archive was built, artifact transfer was performed, new-server
GPU/SQL integration test was run, or complete experiment was rerun. The pack,
transfer and restore tools make those follow-up steps explicit. Source exports
are a point-in-time copy; active campaigns can continue changing their live
status/results after this capture.
