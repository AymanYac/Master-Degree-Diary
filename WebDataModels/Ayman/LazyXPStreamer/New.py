class NFA(object):
    def __init__(self):
        self.start_edges = []
        self.edges = {}  # from_node_id -> [(to_node_id, test), ...]
        self.terminal_nodes = set()
        # The start node has no type
        self.match_types = {}  # node_id -> match_type
        self.labeled_handlers = {} # node_id -> (label, PushtreeHandler)

    def copy(self):
        nfa = NFA()
        nfa.start_edges[:] = self.start_edges
        nfa.edges.update(self.edges)
        nfa.terminal_nodes.update(self.terminal_nodes)
        nfa.match_types.update(self.match_types)
        nfa.labeled_handlers.update(self.labeled_handlers)
        return nfa

    def get_edges(self, node_id):
        if node_id is None:
            return self.start_edges
        return self.edges[node_id]

    def add_handler(self, labeled_handler):
        for node_id in self.terminal_nodes:
            self.labeled_handlers[node_id].append(labeled_handler)


    def new_node(self, from_node_id, test):
        edges = self.get_edges(from_node_id)
        to_node_id = next(counter)
        self.edges[to_node_id] = []
        self.match_types[to_node_id] = test.match_type
        self.labeled_handlers[to_node_id] = []
        edges.append( (to_node_id, test) )
        return to_node_id

    def connect(self, from_node_id, to_node_id, test):
        self.get_edges(from_node_id).append( (to_node_id, test) )

    def extend(self, other):
        assert not set(self.edges) & set(other.edges), "non-empty intersection"
        if not self.start_edges:
            self.start_edges[:] = other.start_edges
        self.edges.update(other.edges)
        self.match_types.update(other.match_types)
        for node_id in self.terminal_nodes:
            self.edges[node_id].extend(other.start_edges)
        self.terminal_nodes.clear()
        self.terminal_nodes.update(other.terminal_nodes)
        self.labeled_handlers.update(other.labeled_handlers)

    def union(self, other):
        assert not set(self.edges) & set(other.edges), "non-empty intersection"
        self.start_edges.extend(other.start_edges)
        self.edges.update(other.edges)
        self.match_types.update(other.match_types)
        self.terminal_nodes.update(other.terminal_nodes)
        self.labeled_handlers.update(other.labeled_handlers)

    def dump(self):
        for node_id, edges in [(None, self.start_edges)] + sorted(self.edges.items()):
            if node_id is None:
                node_name = "(start)"
                labels = ""
            else:
                node_name = str(node_id)
                action = str(self.match_types[node_id])
                labels += " " + str([x[0] for x in self.labeled_handlers[node_id]])
            is_terminal = "(terminal)" if (node_id in self.terminal_nodes) else ""
            print node_name, is_terminal, labels
            self._dump_edges(edges)
        print "======"

    def _dump_edges(self, edges):
        for (to_node_id, test) in edges:
            print "", test, "->", to_node_id