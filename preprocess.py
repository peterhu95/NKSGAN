# coding=utf-8
# GAT preprocess
import torch
import os
import numpy as np


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2  # �����ַ���

def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []  # ��Ԫ���б�Ԫ����Ԫ��(ͷʵ��id����ϵid��βʵ��id)

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []  # �������б��ʾһ���ڽӾ��������Ԫ�ض���id��Ԫ�ػ����ظ�
    unique_entities = set()  # �ṹ��["/m/0b76d_m",...]���ڴ洢���в��ظ���ʵ��ԭʼ���ƣ�/m/0b76d_m��
    for line in lines:
        e1, relation, e2 = parse_line(line)  # �ó�ÿһ�е�ͷʵ�塢��ϵ��βʵ��
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:  # ����ͼ�Ļ��������У�ͷ��ָ���У�β����Ҳ���У�ͷ��ָ���У�β��
                # Connecting source and tail entity
            rows.append(entity2id[e1])  # ͷʵ���id�ŵ����к�
            cols.append(entity2id[e2])  # βʵ���id�ŵ����к�
            if is_unweigted:
                data.append(1)  # ��Ȩ��ͼ���ڽӾ����е�Ԫ��1����ʾͷʵ���βʵ��������
            else:
                data.append(relation2id[relation])  # Ĭ������Ȩ��ͼ���ڽӾ����е�Ԫ���ǹ�ϵ��id

        # Connecting tail and source entity  ����ͼ�Ļ���ֻ���У�ͷ��ָ���У�β��
        rows.append(entity2id[e2])  # βʵ���id�ŵ����к�
        cols.append(entity2id[e1])  # ͷʵ���id�ŵ����к�
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    # ��ӡ��������train.txt��valid.txt��test.txt��ʵ������ֻ��ĳһ���ļ�����ʵ������������ʵ������
    print("number of unique_entities ->", len(unique_entities))  # ����ѵ�����ݵ�ʵ����������ʵ��������Щʵ��δ��ѵ���г���
    return triples_data, (rows, cols, data), list(unique_entities)

def build_data(kb_index,path='./wn18rr/', is_unweigted=False, directed=True):  # Ĭ������Ȩ������ͼ
    entity2id = kb_index.ent_id  # ��ǰ���ñ���ˡ�entity2id���ֵ�{ʵ�������ַ���: ���id}
    relation2id = kb_index.rel_id
    #print("ent2id",entity2id)
    #print("rel2id",relation2id)

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    # �ֱ��ó�ѵ�����ݡ���֤���ݺͲ�������
    # train_triples���б�Ԫ����Ԫ��(ͷʵ��id����ϵid��βʵ��id)��
    # train_adjacency_mat��Ԫ�飬(�ڽӾ�����[ʵ��id...]����[ʵ��id...]���ڽӾ����е�Ԫ��[��ȨΪ1����ȨΪ��ϵid...])��ʵ��id���ظ�����Ϊ����ʱ����ѵ�����ݵ�ÿһ�У�����Ļ�������(ͷ)ָ����(β)
    # unique_entities_train���б��б���Ԫ�ز��ظ�
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    #id2entity = {v: k for k, v in entity2id.items()}  # ��һ��id2entity���ֵ�{���id: ʵ�������ַ���}
    #id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}  # ���ڴ洢ʵ����Ϊͷʵ���βʵ����ֵĴ���
    # left_entity�ṹ��{��ϵid: {ͷʵ��id: ���ʵ����Ϊ�����ϵ��ͷʵ��ĳ��ִ���}}

    with open(os.path.join(path, 'train.txt')) as f:  # ��ѵ�����ݲ���
        lines = f.readlines()  # lines���б�Ԫ������Ԫ���ַ���

    for line in lines:
        e1, relation, e2 = parse_line(line)  # ��ͷʵ�塢��ϵ��βʵ����ַ����ֱ��ó�

        # Count number of occurences for each (e1, relation)  # ������Ԫ��(e1, relation, e2)��˵
        if relation2id[relation] not in left_entity:  # �����ϵid����left_entity�У���Ҫ���������ϵ�Ŀ��ֵ�
            left_entity[relation2id[relation]] = {}   # left_entity{��ϵid: {},...}
        if entity2id[e1] not in left_entity[relation2id[relation]]:  # ���e1��Ϊͷʵ�壬ȴ����left_entity��relation���ֵ���
            left_entity[relation2id[relation]][entity2id[e1]] = 0    # �򴴽����ʵ��Ŀ��ֵ�
        left_entity[relation2id[relation]][entity2id[e1]] += 1       # ����һ�Σ��������һ

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}  # ����ÿ����ϵƽ����Ӧ�ж���ͷʵ�塣left_entity_avg�ṹ��{��ϵid: �����ϵƽ��ͷʵ����Ŀ}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])  # left_entity[i]��i��ϵ��Ӧ��ͷʵ���ֵ�
                                                                  # left_entity[i].values()��ͷʵ����ִ���

    right_entity_avg = {}  # ����ÿ����ϵƽ����Ӧ�ж���βʵ��
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}  # �ṹ��{��ϵid: 1000��ƽ��βʵ����Ŀ/ƽ��ͷβʵ����Ŀ}����corruptʱ�滻ͷ��βʵ��ĸ���(ֱ��������)������1000���Ŵ�����ֵ��
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train
