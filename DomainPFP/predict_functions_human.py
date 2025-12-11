from argparse import ArgumentParser
from datetime import datetime
import torch
import pickle
import json
import sys, os
sys.path.append(os.path.abspath('DomainPFP'))
import numpy as np
import pandas as pd
from tabulate import tabulate
pd.set_option('display.max_colwidth', None)

DOUBLE_PRECISION = 12       # precision for floating point

from domaingo_embedding_model import DomainGOEmbeddingModel, load_domaingo_embedding_model_weights
from domain_embedding import DomainEmbedding
from download_sequences import download_sequence
from utils import Ontology

parser = ArgumentParser()
parser.add_argument('--protein',  help='Uniprot ID of protein', type=str)
parser.add_argument('--fasta',  help='Fasta file of protein sequence', type=str)
parser.add_argument('--blast_flag', help='Use Blast information to predict functions. (DiamondBlast must be installed with path information)',  action="store_true")
parser.add_argument('--diamond_path', help='Path to Diamond Blast (by default the colab release path is provided)',  default='/content/Domain-PFP/diamond', type=str)
parser.add_argument('--ppi_flag', help='Use PPI information to predict functions. (Only works when a uniprot ID or properly formatted fasta file is provided)', action="store_true")
parser.add_argument('--outfile',  help='Path to the output csv file (optional)', type=str)

args = parser.parse_args()


def parse_domains(domain_in_file):
    domains = []
    for domain in domain_in_file:
        domains.append(domain[0])
    return domains
    

def compute_embeddings(domains,dmn_embedding_mf,dmn_embedding_bp,dmn_embedding_cc):
    """
    Computes the protein embedding from the domains 

    Args:
        domains (set or list): set or list of domains

    Returns:
        tuple of 3 numpy arrays: (MF embedding, BP embedding, CC embedding)
    """

    cnt = 0
    for dmn in domains:
        if dmn_embedding_mf.contains(dmn):
            mf_embedding += dmn_embedding_mf.get_embedding(dmn)
            cnt += 1
    if(cnt>1):
        mf_embedding /= cnt                     # averaging


    cnt = 0
    for dmn in domains:
        if dmn_embedding_bp.contains(dmn):
            bp_embedding += dmn_embedding_bp.get_embedding(dmn)
            cnt += 1
    if(cnt>1):
        bp_embedding /= cnt

    cnt = 0
    for dmn in domains:
        if dmn_embedding_cc.contains(dmn):
            cc_embedding += dmn_embedding_cc.get_embedding(dmn)
            cnt += 1
    if(cnt>1):
        cc_embedding /= cnt


    mf_embedding = np.round(mf_embedding,DOUBLE_PRECISION)
    bp_embedding = np.round(bp_embedding,DOUBLE_PRECISION)
    cc_embedding = np.round(cc_embedding,DOUBLE_PRECISION)

    return mf_embedding, bp_embedding, cc_embedding

def merge_predictions(preds1, preds2, preds3):

    preds = {}
    cnt = 1
    if(len(preds2)>0):
        cnt += 1
    if(len(preds3)>0):
        cnt += 1

    go_trms_all = set(preds1.keys()).union(set(preds2.keys())).union(set(preds3.keys()))

    for go_trm in go_trms_all:

        scr = 0
        if go_trm in preds1:
            scr += preds1[go_trm]
        if go_trm in preds2:
            scr += preds2[go_trm]
        if go_trm in preds3:
            scr += preds3[go_trm]

        scr /= cnt

        preds[go_trm] = scr

    return preds

def predict_functions(mf_embedding, bp_embedding, cc_embedding):
    """
    Predict functions of the protein using KNN

    Args:
        mf_embedding (_type_): _description_
        bp_embedding (_type_): _description_
        cc_embedding (_type_): _description_

    Returns:
        _type_: _description_
    """

    knn_mdl_mf = pickle.load(open(os.path.join('saved_models','knn_netgo_mf.p'),'rb'))
    knn_mdl_bp = pickle.load(open(os.path.join('saved_models','knn_netgo_bp.p'),'rb'))
    knn_mdl_cc = pickle.load(open(os.path.join('saved_models','knn_netgo_cc.p'),'rb'))

    all_protein_domains_mf = pickle.load(open(os.path.join('data','processed','all_protein_domains_netgo_mf_train.p'),'rb'))
    all_protein_domains_bp = pickle.load(open(os.path.join('data','processed','all_protein_domains_netgo_bp_train.p'),'rb'))
    all_protein_domains_cc = pickle.load(open(os.path.join('data','processed','all_protein_domains_netgo_cc_train.p'),'rb'))
    all_protein_gos_mf = pickle.load(open(os.path.join('data','processed','all_protein_go_netgo_mf_train.p'),'rb'))
    all_protein_gos_bp = pickle.load(open(os.path.join('data','processed','all_protein_go_netgo_bp_train.p'),'rb'))
    all_protein_gos_cc = pickle.load(open(os.path.join('data','processed','all_protein_go_netgo_cc_train.p'),'rb'))
    
    prtns_mf = set(all_protein_domains_mf.keys())
    prtns_mf = prtns_mf.intersection(set(all_protein_gos_mf.keys()))
    prtns_mf = list(prtns_mf)
    prtns_mf.sort()
    prtns_bp = set(all_protein_domains_bp.keys())
    prtns_bp = prtns_bp.intersection(set(all_protein_gos_bp.keys()))
    prtns_bp = list(prtns_bp)
    prtns_bp.sort()
    prtns_cc = set(all_protein_domains_cc.keys())
    prtns_cc = prtns_cc.intersection(set(all_protein_gos_cc.keys()))
    prtns_cc = list(prtns_cc)
    prtns_cc.sort()

    go_terms_mf = []
    for prtn in prtns_mf:
        if(len(all_protein_domains_mf[prtn])==0):
            continue
        go_terms_mf.append(all_protein_gos_mf[prtn])
    go_terms_bp = []
    for prtn in prtns_bp:
        if(len(all_protein_domains_bp[prtn])==0):
            continue
        go_terms_bp.append(all_protein_gos_bp[prtn])
    go_terms_cc = []
    for prtn in prtns_cc:
        if(len(all_protein_domains_cc[prtn])==0):
            continue
        go_terms_cc.append(all_protein_gos_cc[prtn])

    go_preds_mf = knn_mdl_mf.get_neighbor_go_terms_proba_batch(go_terms_mf, [mf_embedding])[0]
    go_preds_bp = knn_mdl_bp.get_neighbor_go_terms_proba_batch(go_terms_bp, [bp_embedding])[0]
    go_preds_cc = knn_mdl_cc.get_neighbor_go_terms_proba_batch(go_terms_cc, [cc_embedding])[0]

    return (go_preds_mf,go_preds_bp,go_preds_cc)

import networkx as nx
import queue
def label_propagate(go_list, go_graph:nx.DiGraph):
    vis={}
    q = queue.Queue()
    all_nodes = go_graph.nodes()
    for go in go_list:
        if go in all_nodes:
            q.put(go)
            vis[go]=1
    while not q.empty():
        u = q.get()
        for v in go_graph.successors(u):
            if v not in vis:
                vis[v]=1
                q.put(v)
    return vis.keys()

def label_list_to_tensor(go_list, go_graph:nx.DiGraph):
    n = go_graph.number_of_nodes()
    multihot = torch.zeros([n])
    for go in go_list:
        multihot[go] = 1.0
    return multihot

def probablity_propagate(prob_list, go_graph:nx.DiGraph, topo_order):
    n = go_graph.number_of_nodes()
    prob_vector = torch.zeros([n])
    for go_id,prob in prob_list:
        prob_vector[go_id] = prob
    # DP
    for u in topo_order:
        for v in go_graph.predecessors(u):
            prob_vector[u] = max(prob_vector[u],prob_vector[v])
    return prob_vector

from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score    
def cal_metrics(pred, actual):
    actual = np.array(actual).flatten()
    pred = np.array(pred).flatten()
    fpr, tpr, th = roc_curve(actual, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    precision, recall, pr_thresh = metrics.precision_recall_curve(actual, pred)
    aupr_score = metrics.auc(recall, precision)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    thresh = pr_thresh[np.argmax(f1)]
    fmax = np.max(f1)
    
    with open("temp_data/logs.txt","a") as f:
        print("AUC: "+str(auc_score))
        f.write("AUC: "+str(auc_score)+"\n")
        print("AUPR: "+str(aupr_score))
        f.write("AUPR: "+str(aupr_score)+"\n")
        print("Fmax: "+str(fmax))
        f.write("Fmax: "+str(fmax)+"\n")
        print("thresh: "+str(thresh))
        f.write("thresh: "+str(thresh)+"\n")

def main():
    """
    Predicts the functions of a query protein
    """
                                                                                    
    domain_mapper_mf = pickle.load(open(os.path.join('data','processed','domain_mapper_netgo_mf.p'),'rb'))      # loading the mapper files
    go_mapper_mf = pickle.load(open(os.path.join('data','processed','go_mapper_netgo_mf.p'),'rb'))              
    mdl_mf = DomainGOEmbeddingModel(domain_mapper_mf,go_mapper_mf)                                              # creating a model
    mdl_mf = load_domaingo_embedding_model_weights(mdl_mf, os.path.join('saved_models','netgo_mf'))             # loading the model weights
    dmn_embedding_mf = DomainEmbedding(mdl_mf, domain_mapper_mf)                                                # creating the Domaing Embedding object

    domain_mapper_bp = pickle.load(open(os.path.join('data','processed','domain_mapper_netgo_bp.p'),'rb'))
    go_mapper_bp = pickle.load(open(os.path.join('data','processed','go_mapper_netgo_bp.p'),'rb'))
    mdl_bp = DomainGOEmbeddingModel(domain_mapper_bp,go_mapper_bp)
    mdl_bp = load_domaingo_embedding_model_weights(mdl_bp, os.path.join('saved_models','netgo_bp'))
    dmn_embedding_bp = DomainEmbedding(mdl_bp, domain_mapper_bp)

    domain_mapper_cc = pickle.load(open(os.path.join('data','processed','domain_mapper_netgo_cc.p'),'rb'))
    go_mapper_cc = pickle.load(open(os.path.join('data','processed','go_mapper_netgo_cc.p'),'rb'))
    mdl_cc = DomainGOEmbeddingModel(domain_mapper_cc,go_mapper_cc)
    mdl_cc = load_domaingo_embedding_model_weights(mdl_cc, os.path.join('saved_models','netgo_cc'))
    dmn_embedding_cc = DomainEmbedding(mdl_cc, domain_mapper_cc)
    
    mf_embedding = dmn_embedding_mf.get_embedding(-1)           # Initialize embeddings
    bp_embedding = dmn_embedding_bp.get_embedding(-1)
    cc_embedding = dmn_embedding_cc.get_embedding(-1)
    
    with open('temp_data/HUMAN_test_protein_info.json') as f:
        protein_info = json.load(f)
    with open(os.path.join("temp_data","go_terms.json"),"r") as f:
        go_terms = json.load(f)
    with open(os.path.join("temp_data","go_id.json"),"r") as f:
        go_vocab = json.load(f)
    ns_map = {"biological_process":"bp","cellular_component":"cc","molecular_function":"mf"}
    ns_ids = {"bp":[],"cc":[],"mf":[]}
    for i,go in enumerate(go_terms):
        ns_ids[ns_map[go["namespace"]]].append(i)
    with open(os.path.join("temp_data","go_graph_nx.pkl"),"rb") as f:
        go_graph_nx = pickle.load(f)
    
    topo_order = list(nx.topological_sort(go_graph_nx))
    
    mf_ppi = {}
    bp_ppi = {}
    cc_ppi = {}

    mf_dmnd = {}
    bp_dmnd = {}
    cc_dmnd = {}
    
    actual = []
    pred = []
    actual_ns = {"bp":[],"cc":[],"mf":[]}
    pred_ns = {"bp":[],"cc":[],"mf":[]}
    for pid in tqdm(protein_info):
        
        domains = parse_domains(protein_info[pid]['domain'])
        labels = protein_info[pid]['go']
        go_id_list = []
        for go in labels:
            if go in go_vocab:
                go_id_list.append(go_vocab[go])
        propagated_go_list = list(label_propagate(go_id_list,go_graph_nx))
        labels = label_list_to_tensor(propagated_go_list,go_graph_nx)
        
        print("Computing Embeddings")
        mf_embedding, bp_embedding, cc_embedding = compute_embeddings(domains,dmn_embedding_mf,dmn_embedding_bp,dmn_embedding_cc)

        print("Predicting Functions")
        go_preds_mf, go_preds_bp, go_preds_cc = predict_functions(mf_embedding, bp_embedding, cc_embedding)

        go_preds_mf = merge_predictions(go_preds_mf, mf_dmnd, mf_ppi)
        go_preds_bp = merge_predictions(go_preds_bp, bp_dmnd, bp_ppi)
        go_preds_cc = merge_predictions(go_preds_cc, cc_dmnd, cc_ppi)

        probablity_list = []
        for go_trm in go_preds_mf:
            if go_trm in go_vocab:
                probablity_list.append([go_vocab[go_trm], go_preds_mf[go_trm]])
        for go_trm in go_preds_bp:
            if go_trm in go_vocab:
                probablity_list.append([go_vocab[go_trm], go_preds_bp[go_trm]])
        for go_trm in go_preds_cc:
            if go_trm in go_vocab:
                probablity_list.append([go_vocab[go_trm], go_preds_cc[go_trm]])
        
        prob_vector = probablity_propagate(probablity_list,go_graph_nx,topo_order)
        labels = labels.unsqueeze(0)
        prob_vector = prob_vector.unsqueeze(0)
        
        actual += labels.tolist()
        pred += prob_vector.tolist()
        for ns in ns_ids:
            ids = ns_ids[ns]
            actual_ns[ns] += labels[:,ids].tolist()
            pred_ns[ns] += prob_vector[:,ids].tolist()
    cal_metrics(pred, actual)
    for ns in ns_ids:
        with open("temp_data/logs.txt","a") as f:
            print("Namespace: "+ns)
            f.write("Namespace: "+ns+"\n")
        cal_metrics(pred_ns[ns], actual_ns[ns])

if __name__=='__main__':
    main()
