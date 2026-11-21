# SemanticOPT 20-round budget amendment

Implemented and resumed with batch size 50 and 20 rounds (1000 proposals per task). Peptide target is now 200000; query remains 200000. Other methods are unchanged. Imported initialization observations do not count toward this cap.

All 93 existing task prefixes were verified byte-for-byte unchanged. All 72 active workers report the new budget, and all nine inference replicas are healthy. Eighteen relevant checks passed, including resumed termination at exactly proposal 1000 and prevention of budget overshoot. See completion.json for live deployment evidence, budget_tests.log for the new tests, and checkpoints/ for the pre-change databases.

This is a reduced-budget SemanticOPT result and must be identified as such in comparisons with methods retaining 5000 proposals.
