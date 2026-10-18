"""Small directed-graph checks with no dependency imports or environment access."""


def is_acyclic(edges):
    nodes = {n for edge in edges for n in edge}
    children = {n: set() for n in nodes}
    indegree = {n: 0 for n in nodes}
    for a, b in edges:
        if b not in children[a]:
            children[a].add(b); indegree[b] += 1
    ready = [n for n in nodes if indegree[n] == 0]
    visited = 0
    while ready:
        node = ready.pop(); visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0: ready.append(child)
    return visited == len(nodes)


def has_path(edges, source='A', target='Y'):
    children = {}
    for a, b in edges: children.setdefault(a, set()).add(b)
    seen, stack = set(), [source]
    while stack:
        node = stack.pop()
        if node == target: return True
        if node not in seen:
            seen.add(node); stack.extend(children.get(node, ()))
    return False
