import numpy as np
import pandas as pd
from treelib import Node, Tree
import copy

from utils import binaryproduct

def search_cycles(H, L):
    """
    Search short cycles of length L in Parity check matrix H
    """

    m,n = H.shape
    cycles = []

    for i in range(n):
        print('current is at %d' % i)
        H_tree = Tree()
        node_dict = {'v_nodes': [], 'c_nodes': []}
        node_dict['v_nodes'].append(i)
        H_tree.create_node(tag='V' + str(i), identifier='V' + str(i), data=node_dict)
        root = i
        parent_id = 'V' + str(i)
        parents = [parent_id]

        cycles_row = []

        t = 0
        while t <= L:
            new_parent = []
            if t % 2 == 0:  # current is at variable node
                for parent_id in parents:
                    parent_node = H_tree.get_node(parent_id)
                    col_num = parent_node.data['v_nodes'][-1]
                    col = H[:, col_num]
                    col_index = np.array(np.where(col > 0))
                    col_index = np.squeeze(col_index, axis=0)

                    for j in col_index:
                        last_c_node_id = H_tree.ancestor(parent_id)
                        node_id = parent_id + 'C' + str(j)
                        # if parent_node.is_root() or j != H_tree.get_node(last_c_node_id).data:
                        if parent_node.is_root() or j not in parent_node.data['c_nodes']:
                            new_data = copy.deepcopy(parent_node.data)
                            new_data['c_nodes'].append(j)
                            H_tree.create_node(tag='C' + str(j), parent=parent_id, identifier=node_id, data=new_data)
                            new_parent.append(node_id)

            if t % 2 != 0:  # current is at check node
                for parent_id in parents:
                    parent_node = H_tree.get_node(parent_id)
                    row = H[parent_node.data['c_nodes'][-1], :]
                    row_index = np.array(np.where(row > 0))
                    row_index = np.squeeze(row_index, axis=0)

                    for j in row_index:
                        last_v_node_id = H_tree.ancestor(parent_id)
                        node_id = parent_id + 'V' + str(j)

                        if j not in parent_node.data['v_nodes']:
                            new_data = copy.deepcopy(parent_node.data)
                            new_data['v_nodes'].append(j)
                            H_tree.create_node(tag='V' + str(j), parent=parent_id, identifier=node_id, data=new_data)
                            new_parent.append(node_id)

                        if j == root:
                            new_data = copy.deepcopy(parent_node.data)
                            new_data['v_nodes'].append(j)
                            if len(new_data['v_nodes']) > 2:
                                cycles_row.append(new_data['v_nodes'])

                                # path = path2root(node_id, H_tree)
                                # print(path)


            t = t + 1
            parents = new_parent
        cycles.append(cycles_row)
        cycles_nodes = []
        for i in range(len(cycles)):
            cycle_temp = []
            for elem in cycles[i]:
                for num in elem:
                    cycle_temp.append(num)
            cycle_temp = np.unique(cycle_temp)
            cycles_nodes.append(cycle_temp)

        file_path = 'cycles_info/cycles.csv'
        cycles_nodes_df = pd.DataFrame(cycles_nodes)
        cycles_nodes_df.to_csv(file_path)
    return cycles, cycles_nodes

def path2root(node_identifier, tree):
    path = []
    level = tree.level(node_identifier)
    n_id = node_identifier
    while level >= 0:
        if tree.get_node(n_id).tag == 'V':
            path.append(tree.get_node(n_id).data)
        parent_id = tree.ancestor(n_id)
        n_id = parent_id
        level = tree.level(n_id)
    return path


def search_trapping_set(cycles, H):
    n_c, n_v = H.shape
    t_set = []
    for row_cycles in cycles:
        for cycle in row_cycles:
            v_bits = np.zeros(n_v)
            for v_node in cycle:
                v_bits[v_node] = 1
            syndrome = binaryproduct(H, v_bits)
            if np.sum(syndrome) != 0:
                cycle = cycle[1:]
                cycle.sort()
                if cycle not in t_set:
                    trap_degree = [len(cycle), np.sum(syndrome)]
                    t_set.append([trap_degree, cycle])    # t_set contains all of cycles in the ldpc
    pd.DataFrame(t_set).to_csv("trap_info.csv", index=False)
    return t_set
