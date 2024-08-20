import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

ct = 0

def graph_e(val):

    def nm(x):
        global ct
        if not hasattr(x, "n_id"):
            ct+=1
            setattr(x, "n_id", ct)
        return x.n_id

    G = nx.DiGraph()

    top = []
    q = [val]
    while len(q) > 0:
        val = q.pop()
        top.append(val)
        G.add_node(nm(val), t=(str(val.value) if val._op=='' else (val._op + " " + str(val.value))) + "\n" + str(val.grad))
        for c in val._in:
            q.append(c)
            G.add_edge(nm(c), nm(val))

    nx.draw(G, graphviz_layout(G, prog='dot'), with_labels=True, node_color='skyblue', font_color='black', labels=nx.get_node_attributes(G, 't'))
    plt.show()
