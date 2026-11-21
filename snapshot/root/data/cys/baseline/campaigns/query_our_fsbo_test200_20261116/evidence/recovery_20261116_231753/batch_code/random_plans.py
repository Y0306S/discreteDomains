def query_proposal(definition, codec, rng):
    from codec.codec import JoinTreeLeaf, JoinTreeBranch, JoinOperator
    trees = [JoinTreeLeaf(table, alias) for table, count in definition.query_tables
             for alias in range(1, count + 1)]
    graph = definition.schema.query_join_graph
    while len(trees) > 1:
        pairs = [(i, j) for i in range(len(trees)) for j in range(i + 1, len(trees))]
        connected = [(i, j) for i, j in pairs if any(
            graph.has_edge(a, b) for a in trees[i].tables_aliases()
            for b in trees[j].tables_aliases())]
        # A disconnected SQL join graph necessarily needs a Cartesian merge.
        i, j = rng.choice(connected or pairs)
        left, right = trees[i], trees[j]
        if rng.randrange(2): left, right = right, left
        branch = JoinTreeBranch(left, right, rng.choice(list(JoinOperator)))
        del trees[j]; del trees[i]; trees.append(branch)
    return dict(plan=codec.encode(trees[0]))

