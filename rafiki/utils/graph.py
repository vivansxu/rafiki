from copy import deepcopy

class CycleException(Exception): pass

class Graph():

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list
        self.topological_order = None

    def get_topological_order(self):

        if self.topological_order:
            return self.topological_order

        adjacency_list = deepcopy(self.adjacency_list)
        queue = self._get_nodes_with_zero_incoming_degrees(self.adjacency_list)
        topological_order = []

        while queue:
            node = queue.pop()
            topological_order.append(node)

            adjacency_list.pop(node, None)
            for node in self._get_nodes_with_zero_incoming_degrees(adjacency_list):
                if node not in queue:
                    queue.append(node)
        
        if adjacency_list:
            raise CycleException
        else:
            self.topological_order = topological_order
            return topological_order

    def _get_nodes_with_zero_incoming_degrees(self, adjacency_list):
        nodes_with_zero_incoming_degrees = set(list(adjacency_list.keys()))
        for node, adjacent_nodes in adjacency_list.items():
            for adjacent_node in adjacent_nodes:
                nodes_with_zero_incoming_degrees.discard(adjacent_node)
        return list(nodes_with_zero_incoming_degrees)


if __name__ == '__main__':
    directed_acyclic_graph = {
        'A': ['B', 'C', 'D'],
        'B': ['E', 'F'],
        'C': ['D', 'F'],
        'D': ['F'],
        'E': ['F'],
        'F': [],
        'G': ['H'],
        'H': []
    }

    graph = Graph(directed_acyclic_graph)
    topological_order = graph.get_topological_order()
    print(topological_order)

