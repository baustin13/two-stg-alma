## The code is partially adapted from https://github.com/RexYing/diffpool




import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable            

import argparse
import os
import pickle
import random
import time

import encoders as encoders
import gen.feat as featgen
from graph_sampler import GraphSampler
import load_data
from coarsen_pooling_with_last_eigen_padding import Graphs as gp
import graph 
import time
import pickle
import cross_val

import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
from tripletnet import tripletnet
from triplet_sampler import TripletSampler

from sklearn.neighbors import KNeighborsClassifier

from networks import Net





# evaluate is based on kNN (on validation and test sets)
def evaluate(train_dataset, val_dataset, model, args, name='Validation', max_num_examples=None, writer=False, iteration=0):
    model.eval()
    

    train_labels = []
    train_embeddings = []

    # this could be validation or test set, but name of variables are 'val_...'
    val_labels = []
    val_embeddings = []

    # for taxi only
    '''    
    embeddings_val = dict()
    embeddings_train = dict()

    for i in range(7):
        embeddings_val[i] = dict()
        embeddings_val[i][0] = []
        embeddings_val[i][1] = []

        embeddings_train[i] = dict()
        embeddings_train[i][0] = []
        embeddings_train[i][1] = []
    '''

     
    for c in train_dataset.keys():
        for graph in train_dataset[c]:
            train_labels.append(graph.graph['label'])
            adj = Variable(torch.Tensor([graph.graph['adj']]), requires_grad=False).cuda()
            h0 = Variable(torch.Tensor([graph.graph['feats']]), requires_grad=False).cuda()
            batch_num_nodes = np.array([graph.graph['num_nodes']])
            assign_input = Variable(torch.Tensor(graph.graph['assign_feats']), requires_grad=False).cuda()

            feat = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            
            learned_feat = feat[0].cpu().data.numpy()
            
            train_embeddings.append(learned_feat)
            if writer == True:
                f.write(str(learned_feat) + '\n')
                h.write(str(graph.graph['label']) + '\n')

            # for taxi only
            '''
            type = graph.graph['type']
            if name == 'Validation' and iteration == 4:
                type = graph.graph['type']
                embeddings_train[type][0].append(learned_feat[0])
                embeddings_train[type][1].append(learned_feat[1])
            '''

    train_size = len(train_embeddings)

    for c in val_dataset.keys():
        for graph in val_dataset[c]:
            val_labels.append(graph.graph['label'])
            adj = Variable(torch.Tensor([graph.graph['adj']]), requires_grad=False).cuda()
            h0 = Variable(torch.Tensor([graph.graph['feats']]), requires_grad=False).cuda()
            batch_num_nodes = np.array([graph.graph['num_nodes']])
            assign_input = Variable(torch.Tensor(graph.graph['assign_feats']), requires_grad=False).cuda()

            feat = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            learned_feat = feat[0].cpu().data.numpy()
            val_embeddings.append(learned_feat)
            if writer == True:
                g.write(str(learned_feat) + '\n')
                k.write(str(graph.graph['label']) + '\n')
            # for taxi only
            '''
            type = graph.graph['type']
            if name == 'Validation' and iteration == 4:
                type = graph.graph['type']
                embeddings_val[type][0].append(learned_feat[0])
                embeddings_val[type][1].append(learned_feat[1])
            '''
    
    val_size = len(val_embeddings)
        
    t0 = time.time()
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_embeddings, train_labels)
    t1 = time.time()
    train_preds = neigh.predict(train_embeddings)
    t2 = time.time()
    val_preds = neigh.predict(val_embeddings)
    t3 = time.time()
    train_labels = np.hstack(train_labels)
    val_labels = np.hstack(val_labels)
    
    result = {'prec': metrics.precision_score(val_labels, val_preds, average='macro'),
              'recall': metrics.recall_score(val_labels, val_preds, average='macro'),
              'acc': metrics.accuracy_score(val_labels, val_preds),
              'F1': metrics.f1_score(val_labels, val_preds, average="micro"),
              'train acc': metrics.accuracy_score(train_labels, train_preds)}
    #print("Time to fit " + str(train_size) + " train graphs: " + str(t1-t0))
    #print("Time to label " + str(train_size) + " train graphs: " + str(t2-t1))
    #print("Time to label " + str(val_size) + " val graphs: " + str(t3-t2))
    

    print("train accuracy: ", result['train acc'])
    #print("val accuracy: ", result['acc'])
    
    if writer == True:
        f.close()
        g.close()
        h.close()
        k.close()

    '''
    if name == 'Validation' and iteration == 4:
        with open(args.task + "_val_128.pkl", 'wb') as z:
            pickle.dump(embeddings_val, z)

        with open(args.task + "_train_128.pkl", 'wb') as t:
            pickle.dump(embeddings_train, t)
    '''

    return result


def train(dataset, model, ensemble_model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True, iteration = 0):
    writer_batch_idx = [0, 3, 6, 9]
    #print("Number of training graphs: ", len(dataset[0]) + len(dataset[1]))
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    print("model parameters")
    print(model.parameters())
    
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    
    best_test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []
    val_accs = []
    val_epochs = []
    test_accs = []
    test_epochs = []
    

    # Triplet Sampler of train graphs
    tripletsampler_tr = TripletSampler(dataset)

    # Triplet Net
    TNet = tripletnet(model)

    # Triplet loss function
    criterion = torch.nn.MarginRankingLoss(margin = args.alpha)

    
    for epoch in range(args.num_epochs):
        
        tripletsampler_tr.shuffle()        
       
        total_time = 0
        avg_loss = 0.0
        pretrain_loss = 0.0
        
        iter = 0

        model.train()
        #print('Epoch: ', epoch)
        while not tripletsampler_tr.end():
            begin_time = time.time()
            model.zero_grad()

            if args.triplet == 'exhaust':
                one_triplet = tripletsampler_tr.exhaust_sample()
            else:
                one_triplet = tripletsampler_tr.sampler()


            dist_p, dist_n, embed_a, embed_p, embed_n = TNet(one_triplet['anchor'], one_triplet['pos'], one_triplet['neg'])  
                        
            target = torch.FloatTensor(dist_p.size()).fill_(-1)
            target = Variable(target).cuda()
            L2_loss = args.l2_regularize * (embed_a.norm(2) + embed_p.norm(2) + embed_n.norm(2))
            L1_loss = args.l1_regularize * (embed_a.norm(1) + embed_p.norm(1) + embed_n.norm(1))
            C_loss = 0           
            
            if args.w > 0:
                X = torch.cat((embed_a, embed_p))
                X = torch.cat((X, embed_n))
                X = torch.transpose(X,0,1)
                row_means = torch.mean(X, 1)
                row_means = torch.reshape(row_means, (args.output_dim, 1))
                X_ = X - row_means
                
                X_T = torch.transpose(X_,0,1)
                
                K = torch.mm(X_, X_T) 
                K = K/ 3
                C = torch.zeros([args.output_dim, args.output_dim])
                for i in range(args.output_dim):
                    C[i][i] = math.exp(-i**2 / args.gamma ** 2)
                C_loss += args.w * torch.norm(C)            

            loss = criterion(dist_p, dist_n, target) + L2_loss + L1_loss + C_loss
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss.item()
            pretrain_loss += loss.item()
            
            
            #print('Iter: ', iter, ', pos dist: {0:.5f}'.format(dist_p.cpu().data[0].item()), ' neg_dist: {0:.5f}'.format(dist_n.cpu().data[0].item()),  ', loss: {0:.5f}'.format(loss.cpu().data.item()))
            elapsed = time.time() - begin_time
            total_time += elapsed
                                        
        avg_loss /= iter 
        pretrain_loss /= iter
        print("average training loss after pre-train at epoch " + str(epoch) + " is: " + str(pretrain_loss))

        

        
    # now, train 1 more time with the final mapping layer
    # Triplet Sampler of train graphs
    tripletsampler_tr = TripletSampler(dataset)
    
    
    in_feat = model.map_model.in_features
    pred_layers = []
    pred_layers.append(nn.Linear(in_feat, 64).cuda())
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(64, 32).cuda())
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(32, 2).cuda())
    pred_model = nn.Sequential(*pred_layers)
   
    model.map_model = pred_model
    optimizer_2 = torch.optim.Adam(model.parameters(), lr=0.001)
        
    result = evaluate(dataset, dataset, model, args, name='Train', max_num_examples=100)
    train_acc = result['acc']
    print("train acc before post-train : " + str(train_acc))

    max_train_acc = 0
    for epoch in range(args.num_epochs * 2):
        posttrain_loss = 0.0
        iter = 0
        
        tripletsampler_tr.shuffle()              
      
        model.train()
        
        while not tripletsampler_tr.end():
            begin_time = time.time()
            #ensemble_model.zero_grad()

            # zero the parameter gradients
            optimizer_2.zero_grad()

            if args.triplet == 'exhaust':
                one_triplet = tripletsampler_tr.exhaust_sample()
            else:
                one_triplet = tripletsampler_tr.sampler()

            anchor_graph = one_triplet['anchor']
            
            label = Variable(torch.LongTensor([int(anchor_graph.graph['label'])])).cuda()
            
            adj = Variable(torch.Tensor([anchor_graph.graph['adj']]), requires_grad=False).cuda()
            h0 = Variable(torch.Tensor([anchor_graph.graph['feats']]), requires_grad=False).cuda()
            batch_num_nodes = np.array([anchor_graph.graph['num_nodes']])
            assign_input = Variable(torch.Tensor(anchor_graph.graph['assign_feats']), requires_grad=False).cuda()

            pred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            
            lamda = torch.tensor(0.).cuda()
            l2_reg = torch.tensor(0.).cuda()
            for param in model.parameters():
                l2_reg += torch.norm(param).cuda()

            loss = F.cross_entropy(F.softmax(pred), label)
            
            posttrain_loss += loss.item()
            iter += 1
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer_2.step()


        posttrain_loss /= iter
        print("training loss at post phase at epoch " + str(epoch) + " is: " + str(posttrain_loss))

        result = evaluate(dataset, dataset, model, args, name='Train', max_num_examples=100)
        train_acc = result['acc']
        print("train acc at epoch " + str(epoch) + " post-train: " + str(train_acc))
        if train_acc > max_train_acc:
            max_train_acc = train_acc


         
    # after training twice, evaluate on validation and test set
    result = evaluate(dataset, dataset, model, args, name='Train', max_num_examples=100)
    train_acc = result['acc']
    print("train acc: " + str(train_acc))
                
    val_acc = 0
    if val_dataset is not None:
        val_result = evaluate(dataset, val_dataset, model, args, name='Validation', iteration=iteration)
        val_acc = val_result['acc']
    if test_dataset is not None:
        test_result = evaluate(dataset, test_dataset, model, args, name='Test', iteration=iteration)
        test_acc = test_result['acc']

       
    return max_train_acc, test_acc, val_acc



def prepare_data(graphs, graphs_list, args, test_graphs=None, max_nodes=0, seed=0):

    zip_list = list(zip(graphs,graphs_list))
    random.Random(seed).shuffle(zip_list)
    graphs, graphs_list = zip(*zip_list)
    #print('Test ratio: ', args.test_ratio)
    #print('Train ratio: ', args.train_ratio)
    test_graphs_list = []

    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1-args.test_ratio))
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
        train_graphs_list = graphs_list[:train_idx]
        val_graphs_list = graphs_list[train_idx: test_idx]
        test_graphs_list = graphs_list[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        train_graphs_list = graphs_list[:train_idx]    
        val_graphs = graphs[train_idx:]    
        val_graphs_list = graphs_list[train_idx: ] 
    #print('Num training graphs: ', len(train_graphs),'; Num validation graphs: ', len(val_graphs),'; Num testing graphs: ', len(test_graphs))

    #print('Number of graphs: ', len(graphs))
    #print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))

    '''
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
    '''
    test_dataset_loader = []
 
    dataset_sampler = GraphSampler(train_graphs,train_graphs_list, args.num_pool_matrix,args.num_pool_final_matrix,normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type, norm = args.norm)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, val_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type, norm = args.norm)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)
    if len(test_graphs)>0:
        dataset_sampler = GraphSampler(test_graphs, test_graphs_list, args.num_pool_matrix, args.num_pool_final_matrix,normalize=False, max_num_nodes=max_nodes,
                features=args.feature_type, norm = args.norm)
        test_dataset_loader = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=args.num_workers)



    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim


def benchmark_task_val(args, feat='node-label', pred_hidden_dims = [50]):

    all_vals = []
    all_tests = []

    data_out_dir = 'data/data_preprocessed/' + args.bmname + '/pool_sizes_' + args.pool_sizes 
    if args.normalize ==0:
        data_out_dir = data_out_dir + '_nor_' + str(args.normalize)


    data_out_dir = data_out_dir + '/'
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    graph_list_file_name = data_out_dir + 'graphs_list.p' 
    dataset_file_name = data_out_dir + 'dataset.p'

    # load the pre-processed datasets or do pre-processin
    if os.path.isfile(graph_list_file_name) and os.path.isfile(dataset_file_name):
        print('Files exist, reading from stored files....')
        #print('Reading file from', data_out_dir)
        with open(dataset_file_name, 'rb') as f:
            graphs = pickle.load(f)
        with open(graph_list_file_name, 'rb') as f:
            graphs_list = pickle.load(f)
        #print('Data loaded!')
    else:
        print('No files exist, preprocessing datasets...')


        graphs = load_data.read_graphfile(args.datadir,args.bmname, max_nodes =args.max_nodes)
        #print('Data length before filtering: ', len(graphs))

        dataset_copy = graphs.copy()

        len_data = len(graphs)
        graphs_list = []
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        #print('pool_sizes: ', pool_sizes )

        for i in range(len_data):

            adj = nx.adjacency_matrix(dataset_copy[i])
            
            # print('Adj shape',adj.shape)
            if adj.shape[0] < args.min_nodes or adj.shape[0]> args.max_nodes or adj.shape[0]!= dataset_copy[i].number_of_nodes():
                graphs.remove(dataset_copy[i])
                
            else:
                
                number_of_nodes = dataset_copy[i].number_of_nodes()

                coarsen_graph = gp(adj.todense().astype(float), pool_sizes)
                # if args.method == 'wave':
                coarsen_graph.coarsening_pooling(args.normalize)

                graphs_list.append(coarsen_graph)


        #print('Data length after filtering: ', len(graphs), len(graphs_list))
        #print('Dataset preprocessed, dumping....')
        with open(dataset_file_name, 'wb') as f:
            pickle.dump(graphs, f)
        with open(graph_list_file_name, 'wb') as f:
            pickle.dump(graphs_list, f)

        #print('Dataset dumped!')


    
    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']

    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])

    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(args.num_iterations):

        train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim = \
                        prepare_data(graphs, graphs_list, args, test_graphs = None,max_nodes=args.max_nodes, seed = i)
            
                       
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        model = Net(args).to(args.device)

        _, test_acc, val_acc = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset)
        all_vals.append(val_acc)
        all_tests.append(test_acc)

    # Report performance on validation
    all_vals_mean = np.mean(all_vals, axis=0)
    all_vals_std = np.std(all_vals, axis=0)
    print("Validation performance " + str(args.num_iterations) + " times: " + str(all_vals_mean) + " with std " + str(all_vals_std))
    
    # Report performance on test
    all_tests_mean = np.mean(all_tests, axis=0)
    all_tests_std = np.std(all_tests, axis=0)
    print("Test performance " + str(args.num_iterations) + " times: " + str(all_tests_mean) + " with std " + str(all_tests_std))



def attack_darpa_task_val(args, writer=None, feat='learnable'):
    all_vals = []
    all_tests = []
    all_trains = []

    if args.datadir == 'attack':
        print('attack data')
        graphs = load_data.read_attackgraph(args.datadir, args.top, max_nodes=args.max_nodes)
    elif args.datadir == 'darpa' or args.datadir == 'taxi':
        print(args.task + ' data')
        graphs = load_data.read_supplementarygraph(args.datadir, args.task, max_nodes=args.max_nodes)

    data_out_dir = 'data/data_preprocessed/' + args.bmname + '/pool_sizes_' + args.pool_sizes 
    if args.normalize ==0:
        data_out_dir = data_out_dir + '_nor_' + str(args.normalize)


    data_out_dir = data_out_dir + '/'
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)


    graph_list_file_name = data_out_dir + 'graphs_list.p' 
    dataset_file_name = data_out_dir + 'dataset.p'

        
    # load the pre-processed datasets or do pre-processin
    if os.path.isfile(graph_list_file_name) and os.path.isfile(dataset_file_name):
        print('Files exist, reading from stored files....')
        #print('Reading file from', data_out_dir)
        with open(dataset_file_name, 'rb') as f:
            graphs = pickle.load(f)
        with open(graph_list_file_name, 'rb') as f:
            graphs_list = pickle.load(f)
        #print('Data loaded!')
    else:
        print('No files exist, preprocessing datasets...')


        graphs = load_data.read_graphfile(args.datadir,args.bmname, max_nodes =args.max_nodes)
        #print('Data length before filtering: ', len(graphs))

        dataset_copy = graphs.copy()

        len_data = len(graphs)
        graphs_list = []
        pool_sizes = [int(i) for i in args.pool_sizes.split('_')]
        #print('pool_sizes: ', pool_sizes )

        for i in range(len_data):

            adj = nx.adjacency_matrix(dataset_copy[i])
            
            # print('Adj shape',adj.shape)
            if adj.shape[0] < args.min_nodes or adj.shape[0]> args.max_nodes or adj.shape[0]!= dataset_copy[i].number_of_nodes():
                graphs.remove(dataset_copy[i])
                
            else:
                
                number_of_nodes = dataset_copy[i].number_of_nodes()

                coarsen_graph = gp(adj.todense().astype(float), pool_sizes)
                # if args.method == 'wave':
                coarsen_graph.coarsening_pooling(args.normalize)

                graphs_list.append(coarsen_graph)


        #print('Data length after filtering: ', len(graphs), len(graphs_list))
        #print('Dataset preprocessed, dumping....')
        with open(dataset_file_name, 'wb') as f:
            pickle.dump(graphs, f)
        with open(graph_list_file_name, 'wb') as f:
            pickle.dump(graphs_list, f)

        #print('Dataset dumped!')


    if args.feature_type == 'learnable':
        # input node features as learnable features
        node_features = torch.empty(args.max_nodes, args.input_dim, requires_grad = True).type(torch.FloatTensor).cuda()
        nn.init.normal_(node_features, std = 2)

        print('Using learnable features')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = node_features[u]

    elif args.feature_type == 'node-labels':
        print('Using node labels as features')
        for G in graphs:
            for u in G.nodes():
                G.nodes[u]['feat'] = np.array(G.nodes[u]['label'])


    for i in range(args.num_iterations):
        train_dataset, test_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
                cross_val.split_train_val_normal(graphs, args, i, max_nodes=args.max_nodes, feat = 'learnable')
        
        
        print('Method: SAGPooling GCN')
        model = Net(args).to(args.device)

                    
        if args.val == True:
            _, train_acc, test_acc, val_acc = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
            writer=writer)
            all_vals.append(val_acc)
            all_tests.append(test_acc)
            all_trains.append(train_acc)
        else:
            _, train_acc, test_acc, val_acc = train(train_dataset, model, args, val_dataset=None, test_dataset=test_dataset,
            writer=writer)
            all_tests.append(test_acc)
            all_trains.append(train_acc)
    
    # Report performance on train
    all_trains_mean = np.mean(all_trains, axis=0)
    all_trains_std = np.std(all_trains, axis=0)
    print("Training performance " + str(args.num_iterations) + " times: " + str(all_trains_mean) + " with std " + str(all_trains_std))

    # Report performance on validation
    if args.val == True:
        all_vals_mean = np.mean(all_vals, axis=0)
        all_vals_std = np.std(all_vals, axis=0)
        print("Validation performance " + str(args.num_iterations) + " times: " + str(all_vals_mean) + " with std " + str(all_vals_std))
    
    # Report performance on test
    all_tests_mean = np.mean(all_tests, axis=0)
    all_tests_std = np.std(all_tests, axis=0)
    print("Test performance " + str(args.num_iterations) + " times: " + str(all_tests_mean) + " with std " + str(all_tests_std))

            
                 

def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    parser.add_argument('--task', dest='task',
            help='Name of the taxi dataset')

    # whether to have a validation sets
    parser.add_argument('--val', dest='val', type=bool,
                        help='Whether to have a validation set')

    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--iterations', dest='num_iterations', type=int,
            help='Number of iterations in benchmark_test_val')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--test-ratio', dest='test_ratio', type=float,
            help='Ratio of number of graphs testing set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')


    parser.add_argument('--pool_sizes', type = str,
                        help = 'pool_sizes', default = '10')
    parser.add_argument('--num_pool_matrix', type =int,
                        help = 'num_pooling_matrix', default = 1)
    parser.add_argument('--min_nodes', type = int,
                        help = 'min_nodes', default = 12)

    parser.add_argument('--weight_decay', type = float,
                        help = 'weight_decay', default = 0.0)
    parser.add_argument('--num_pool_final_matrix', type = int,
                        help = 'number of final pool matrix', default = 0)

    parser.add_argument('--normalize', type = int,
                        help = 'nomrlaized laplacian or not', default = 0)
    parser.add_argument('--pred_hidden', type = str,
                        help = 'pred_hidden', default = '50')

    
    parser.add_argument('--num_shuffle', type = int,
                        help = 'total num_shuffle', default = 10)
    parser.add_argument('--shuffle', type = int,
                        help = 'which shuffle, choose from 0 to 9', default=0)
    parser.add_argument('--concat', type = int,
                        help = 'whether concat', default = 1)
    parser.add_argument('--feat', type = str,
                        help = 'which feat to use', default = 'node-label')
    parser.add_argument('--mask', type = int,
                        help = 'mask or not', default = 1)
    parser.add_argument('--norm', type = str,
                        help = 'Norm for eigens', default = 'l2')

    parser.add_argument('--with_test', type = int,
                        help = 'with test or not', default = 0)
    parser.add_argument('--con_final', type = int,
                        help = 'con_final', default = 1)
    parser.add_argument('--cuda', dest = 'cuda',
                        help = 'cuda', default = 0)
    parser.set_defaults(max_nodes=1000,
			task='ppi',
                        feature_type='default',
                        datadir = 'data',
                        val=True,
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=10,
                        num_iterations=5,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=6,
                        num_gc_layers=2,
                        dropout=0.0,
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()
    seed = 1
    print(prog_args)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('bmname: ', prog_args.bmname)
    print('num_classes: ', prog_args.num_classes)
    # print('method: ', prog_args.method)
    print('batch_size: ', prog_args.batch_size)
    print('num_pool_matrix: ', prog_args.num_pool_matrix)
    print('num_pool_final_matrix: ', prog_args.num_pool_final_matrix)
    print('epochs: ', prog_args.num_epochs)
    print('learning rate: ', prog_args.lr)
    print('num of gc layers: ', prog_args.num_gc_layers)
    print('output_dim: ', prog_args.output_dim)
    print('hidden_dim: ', prog_args.hidden_dim)
    print('pred_hidden: ', prog_args.pred_hidden)
    # print('if_transpose: ', prog_args.if_transpose)
    print('dropout: ', prog_args.dropout)
    print('weight_decay: ', prog_args.weight_decay)
    print('shuffle: ', prog_args.shuffle)
    print('Using batch normalize: ', prog_args.bn)
    print('Using feat: ', prog_args.feat)
    print('Using mask: ', prog_args.mask)
    print('Norm for eigens: ', prog_args.norm)
    # print('Combine pooling results: ', prog_args.pool_m)
    print('With test: ', prog_args.with_test)

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)
   

   
    pred_hidden_dims = [int(i) for i in prog_args.pred_hidden.split('_')]
    if prog_args.bmname is not None:
        
        if prog_args.task == 'ppi':
            benchmark_task_val(prog_args, pred_hidden_dims = pred_hidden_dims, feat = prog_args.feat)
        else:
            attack_darpa_task_val(prog_args)



if __name__ == "__main__":
    main()