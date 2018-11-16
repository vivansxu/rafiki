from copy import deepcopy

class InvalidDAGException(Exception): pass

def build_dag(models, ensemble):
    adjacency_list = {}
    for model in models:
        adjacency_list[model['name']] = [] if ensemble is None else [ensemble['name']]
    return adjacency_list

def validate_dag(adjacency_list):
    try:
        _get_topological_order(adjacency_list)
        return True
    except InvalidDAGException:
        return False

def get_parent(model, adjacency_list):
    parents = []
    for node, adjacent_nodes in adjacency_list.items():
        if model in  adjacent_nodes:
            parents.append(node)
    return parents

def _get_topological_order(adjacency_list):
    adjacency_list = deepcopy(adjacency_list)
    queue = _get_nodes_with_zero_incoming_degrees(adjacency_list)
    topological_order = []

    while queue:
        node = queue.pop()
        topological_order.append(node)

        adjacency_list.pop(node, None)
        for node in _get_nodes_with_zero_incoming_degrees(adjacency_list):
            if node not in queue:
                queue.append(node)
    
    if adjacency_list:
        raise InvalidDAGException
    else:
        return topological_order  

def _get_nodes_with_zero_incoming_degrees(adjacency_list):
    nodes_with_zero_incoming_degrees = set(list(adjacency_list.keys()))
    for node, adjacent_nodes in adjacency_list.items():
        for adjacent_node in adjacent_nodes:
            nodes_with_zero_incoming_degrees.discard(adjacent_node)
    return list(nodes_with_zero_incoming_degrees)
