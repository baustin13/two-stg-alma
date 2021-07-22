import os
import re
import sys

import torch
from torch.autograd import Variable

sys.path.append('/tstg_test_dep')
import resolution_prebuffer
import numpy as np
import alma
from alma_utils import *
import random

def res_task_lits(lit_str):
    L = lit_str.split('\n')[:-1]
    return [ x.split('\t') for x in L]

def explosion(size, kb):
    """
    The one axiom we work with:
       if(and(distanceAt(Item1, D1, T), distanceAt(Item2, D2, T)), distanceBetweenBoundedBy(D1, Item1, Item2, T)).
    """
    global alma_inst,res
    ilist = list(range(size))
    random.shuffle(ilist)
    for i in ilist:
        print("i=", i)
        if "test1_kb.pl" in kb:
            obs_fmla_a = "distanceAt(a, {}, {}).".format(random.randint(0, 10), i)
            obs_fmla_b = "distanceAt(b, {}, {}).".format(random.randint(0, 10), i)
            alma.add(alma_inst, obs_fmla_a)
            alma.add(alma_inst, obs_fmla_b)
        elif "january_preglut.pl" in kb:
            obs_fmla = "location(a{}).".format(i)
            alma.add(alma_inst, obs_fmla)
        elif "qlearning1.pl" in kb:
            # obs_fmla_a = "f({}).".format(i)
            obs_fmla_b = "g({}).".format(i)
            # alma.add(alma_inst, obs_fmla_a)
            alma.add(alma_inst, obs_fmla_b)
        # elif "ps_test_search.pl" in kb:
        r = alma.prebuf(alma_inst)
        alma.astep(alma_inst)
    print("Explosion done.")
    print("="*80)
    return r


def two_stg_test(network, network_priors, exp_size=10, num_steps=500, alma_heap_print_size=100, prb_print_size=30, numeric_bits=10, heap_print_freq=10, prb_threshold=-1, use_gnn = True, kb='/home/justin/alma-2.0/glut_control/test1_kb.pl', gnn_nodes=2000, initial_test=False):
    global alma_inst,res
    alma_inst,res = alma.init(1,kb, '0', 1, 1000, [], [])
    dbb_instances = []
    exp = explosion(exp_size, kb)
    res_tasks = exp[0]
    if len(res_tasks) == 0:
        return []
    res_lits = res_task_lits(exp[2])
    #subjects = ['a0', 'a1', 'b0', 'b1']
    # Compute initial subjects.  We want 'a','b' and first 8K integers in each of three places
    subjects = []
    for place in range(3):
        for cat_subj in ['a', 'b']:
            subjects.append("{}/{}".format(cat_subj, place))
        for num_subj in range(2 ** numeric_bits):
            subjects.append("{}/{}".format(num_subj, place))
    # for place in range(3):
    #     for cat_subj in ['a', 'b']:
    #         subjects.append("{}/{}".format(cat_subj, place))
    #     for num_subj in range(2**numeric_bits):
    #         subjects.append("{}/{}".format(num_subj, place))

    subjects = ['a', 'b', 'distanceAt', 'distanceBetweenBoundedBy']
    for place in range(3):
        for cat_subj in ['a', 'b']:
            subjects.append("{}/{}".format(cat_subj, place))


    # network.eval()
    res_task_input = [ x[:2] for x in res_tasks]
    if network_priors:
        temp_network = resolution_prebuffer.res_prebuffer(subjects, [], use_gnn=use_gnn, gnn_nodes=gnn_nodes)

        prb = alma.prebuf(alma_inst)
        res_tasks = prb[0]
        res_lits = res_task_lits(prb[2])
        res_task_input = [x[:2] for x in res_tasks]
        temp_network.save_batch(res_task_input, res_lits)
        X = temp_network.Xbuffer
        Y = temp_network.ybuffer

        two_stg_dataset(X, Y)
        dataset = read_graphfile('2STGTest', 'ALMA')

        for c in dataset.keys():
            for graph in dataset[c]:
                adj = Variable(torch.Tensor([graph.graph['adj']]), requires_grad=False).cuda()
                h0 = Variable(torch.Tensor([graph.graph['feats']]), requires_grad=False).cuda()
                batch_num_nodes = np.array([graph.graph['num_nodes']])
                assign_input = Variable(torch.Tensor(graph.graph['assign_feats']), requires_grad=False).cuda()

                feat, out = model(h0, adj, batch_num_nodes, assign_x=assign_input)
                priorities = np.zeros(len(X))
                for i in range(len(priorities)):
                    # priorities[i] = pred[i][1]                      # activation val
                    # priorities[i] = 1 / (1 + np.exp(priorities[i]))  # sigmoid for cross entropy
                    p0 = float()
                    p1 = float()
                    priorities[i] = 1 - (np.exp(p1)/(np.exp(p1) + np.exp(p0)))
    else:
        priorities = np.random.uniform(size=len(res_task_input))

    alma.set_priors_prb(alma_inst, priorities.flatten().tolist())
    alma.prb_to_res_task(alma_inst, prb_threshold)

    #print("prb: ", alma.prebuf(alma_inst))
    kb = alma.kbprint(alma_inst)[0]
    print("kb: ")
    for s in kb.split('\n'):
        print(s)
    for idx in range(num_steps):
        prb = alma.prebuf(alma_inst)[0]
        if (idx % heap_print_freq == 0):
            print("Step: ", idx)
            print("prb size: ", len(prb))
            for fmla in prb:
                print(fmla)
            print("\n"*3)
            print("KB:")
            for fmla in alma.kbprint(alma_inst)[0].split('\n'):
                print(fmla)
                if ': distanceBetweenBoundedBy' in fmla:
                    dbb_instances.append(fmla)
            print("DBB {}: {}".format(idx, len(dbb_instances)))

            rth = alma.res_task_buf(alma_inst)
            print("Heap:")
            print("HEAP size {}: {} ".format(idx, len(rth[1].split('\n')[:-1])))
            for i, fmla in enumerate(rth[1].split('\n')[:-1]):
                pri = rth[0][i][-1]
                print("i={}:\t{}\tpri={}".format(i, fmla, pri))
                if i >  alma_heap_print_size:
                    break
            print("-"*80)

        if len(prb) > 0:
            res_task_input = [x[:2] for x in prb]

            if network_priors:
                temp_network = resolution_prebuffer.res_prebuffer(subjects, [], use_gnn=use_gnn, gnn_nodes=gnn_nodes)

                prb = alma.prebuf(alma_inst)
                res_tasks = prb[0]
                res_lits = res_task_lits(prb[2])
                res_task_input = [x[:2] for x in res_tasks]
                temp_network.save_batch(res_task_input, res_lits)
                X = temp_network.Xbuffer
                Y = temp_network.ybuffer

                # PRIORS AGAIN HERE

            else:
                priorities = np.random.uniform(size=len(res_task_input))

            alma.set_priors_prb(alma_inst, priorities.flatten().tolist())
            alma.prb_to_res_task(alma_inst, prb_threshold)
        #alma.add(alma_inst, "distanceAt(a, {}, {}).".format(idx, idx))
        alma.astep(alma_inst)
    return dbb_instances




def two_stg_dataset(X, Y):
    dim_nfeats = len(X[0][1][0])

    gclasses = 2
    NODES_PER_GRAPH = 100

    num_nodes = 0
    num_edges = 0
    num_graphs = 0

    if not os.path.exists('2STGTest'):
        os.mkdir('2STGTest')

    edges_file = open('2STGTest/ALMA_A.txt', 'a')                        # STORE EDGES
    graph_indicator = open('2STGTest/ALMA_graph_indicator.txt', 'a')     # INDEX NODES TO GRAPHS
    node_labels = open('2STGTest/ALMA_node_labels.txt', 'a')             # ALL ZEROES
    graph_labels = open('2STGTest/ALMA_graph_labels.txt', 'a')           # GRAPH CLASS LABELS
    node_attributes = open('2STGTest/ALMA_node_attributes.txt', 'a')     # NODE FEATURES

    for graph in X:
        y = 1 + (num_graphs * NODES_PER_GRAPH)                       # 1, NOT 0, THAT'S HOW DATA_LOAD EXPECTS
        for row in graph[0]:
            x = 1 + (num_graphs * NODES_PER_GRAPH)
            for column in row:
                if column == 1.0:
                    edges_file.write(str(y) + ', ' + str(x) + '\n')  # WRITE THE EDGES
                    edges_file.write(str(x) + ', ' + str(y) + '\n')
                    num_edges += 2
                x += 1
            y += 1
        edges_file.write(str(1 + (num_graphs * NODES_PER_GRAPH)) + ', ' + str(51 + (num_graphs * NODES_PER_GRAPH)) + '\n')
        edges_file.write(str(51 + (num_graphs * NODES_PER_GRAPH)) + ', ' + str(1 + (num_graphs * NODES_PER_GRAPH)) + '\n')
        num_edges += 2                                          # LINK THE TREES
        for i in range(NODES_PER_GRAPH):
            graph_indicator.write(str(num_graphs + 1) + '\n')       # LABEL NODES TO GRAPHS THEY BELONG WITH
            node_labels.write('1\n')                            # FILL NODE LABELS WITH NOTHING (ACTUAL NODE DATA IN NODE_ATTRIBUTES)
        for features in graph[1]:
            ei = 0
            for ei in range(len(features) - 1):
                node_attributes.write(str(int(features[ei])) + ', ')      # WRITE NODE FEATURES
            node_attributes.write(str(int(features[ei + 1])))       # NO TRAILING COMMA
            node_attributes.write('\n')
        num_graphs += 1                                         # ITERATE NUM_GRAPHS AND NUM_NODES
        num_nodes += NODES_PER_GRAPH

        # sys.exit()                                            # TEST ON ONE GRAPH

    for label in Y:
        graph_labels.write(str(int(label)) + '\n')  # WRITE GRAPH CLASS LABEL

    edges_file.close()
    graph_indicator.close()
    node_labels.close()
    graph_labels.close()
    node_attributes.close()


# (2STGTest, ALMA)
def read_graphfile(datadir, dataname, max_nodes=None):
    '''
    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    # assume that all graph labels appear in the dataset
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            # if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    # graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # if label_has_zero:
    #    graph_labels += 1

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        # print("graph label:")
        # print(graph_labels[i-1])
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                G.nodes[u]['label'] = node_label_one_hot
                G.nodes[u]['Label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.nodes[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        # if float(nx.__version__)<2.0:     # Broken on networkx 2.5.1
        if False:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1

        # indexed from 0
        chance = random.random()

        graphs.append(nx.relabel_nodes(G, mapping))
    # size = math.floor(len(graphs) / 9)
    return graphs

model = torch.load('saved_models/model0.pt')
model.eval()
two_stg_test(model, True)