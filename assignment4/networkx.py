#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx

# Create a networkx graph object
my_graph = nx.Graph() 

# Add edges to to the graph object
# Each tuple represents an edge between two nodes
my_graph.add_edges_from([
                        (2,3), 
                        (1,3), 
                        (3,4), 
                        (1,5), 
                        (3,5),
                        (4,2),
                        (2,3),
                        (3,0)])

# Draw the resulting graph
nx.draw(my_graph, with_labels=True, font_weight='bold',title="small example of networkX")
