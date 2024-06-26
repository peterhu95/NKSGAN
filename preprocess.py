# coding=utf-8
# GAT preprocess
import torch
import os
import numpy as np


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2  # 返回字符串

def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []  # 三元组列表，元素是元组(头实体id，关系id，尾实体id)

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []  # 用三个列表表示一个邻接矩阵，里面的元素都是id，元素会有重复
    unique_entities = set()  # 结构：["/m/0b76d_m",...]用于存储所有不重复的实体原始名称（/m/0b76d_m）
    for line in lines:
        e1, relation, e2 = parse_line(line)  # 拿出每一行的头实体、关系、尾实体
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:  # 无向图的话，既有行（头）指向列（尾），也有列（头）指向行（尾）
                # Connecting source and tail entity
            rows.append(entity2id[e1])  # 头实体的id号当做行号
            cols.append(entity2id[e2])  # 尾实体的id号当做列号
            if is_unweigted:
                data.append(1)  # 无权重图，邻接矩阵中的元素1，表示头实体和尾实体有连边
            else:
                data.append(relation2id[relation])  # 默认是有权重图，邻接矩阵中的元素是关系的id

        # Connecting tail and source entity  有向图的话，只有列（头）指向行（尾）
        rows.append(entity2id[e2])  # 尾实体的id号当做行号
        cols.append(entity2id[e1])  # 头实体的id号当做列号
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    # 打印出来的是train.txt或valid.txt或test.txt的实体数，只是某一个文件的总实体数（不是总实体数）
    print("number of unique_entities ->", len(unique_entities))  # 例如训练数据的实体数少于总实体数，有些实体未在训练中出现
    return triples_data, (rows, cols, data), list(unique_entities)

def build_data(kb_index,path='./wn18rr/', is_unweigted=False, directed=True):  # 默认是有权、有向图
    entity2id = kb_index.ent_id  # 提前做好编号了。entity2id是字典{实体名称字符串: 编号id}
    relation2id = kb_index.rel_id
    #print("ent2id",entity2id)
    #print("rel2id",relation2id)

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    # 分别拿出训练数据、验证数据和测试数据
    # train_triples是列表，元素是元组(头实体id，关系id，尾实体id)，
    # train_adjacency_mat是元组，(邻接矩阵行[实体id...]，列[实体id...]，邻接矩阵中的元素[无权为1，有权为关系id...])。实体id有重复，因为构建时遍历训练数据的每一行，有向的话，从列(头)指向行(尾)
    # unique_entities_train是列表，列表中元素不重复
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    #id2entity = {v: k for k, v in entity2id.items()}  # 做一个id2entity的字典{编号id: 实体名称字符串}
    #id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}  # 用于存储实体作为头实体或尾实体出现的次数
    # left_entity结构：{关系id: {头实体id: 这个实体作为这个关系的头实体的出现次数}}

    with open(os.path.join(path, 'train.txt')) as f:  # 对训练数据操作
        lines = f.readlines()  # lines是列表，元素是三元组字符串

    for line in lines:
        e1, relation, e2 = parse_line(line)  # 将头实体、关系、尾实体的字符串分别拿出

        # Count number of occurences for each (e1, relation)  # 对于三元组(e1, relation, e2)来说
        if relation2id[relation] not in left_entity:  # 如果关系id不在left_entity中，则要创建这个关系的空字典
            left_entity[relation2id[relation]] = {}   # left_entity{关系id: {},...}
        if entity2id[e1] not in left_entity[relation2id[relation]]:  # 如果e1作为头实体，却不在left_entity的relation的字典中
            left_entity[relation2id[relation]][entity2id[e1]] = 0    # 则创建这个实体的空字典
        left_entity[relation2id[relation]][entity2id[e1]] += 1       # 出现一次，则次数加一

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}  # 计算每个关系平均对应有多少头实体。left_entity_avg结构：{关系id: 这个关系平均头实体数目}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])  # left_entity[i]是i关系对应的头实体字典
                                                                  # left_entity[i].values()是头实体出现次数

    right_entity_avg = {}  # 计算每个关系平均对应有多少尾实体
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}  # 结构：{关系id: 1000乘平均尾实体数目/平均头尾实体数目}。做corrupt时替换头、尾实体的概率(直观意义上)。乘了1000，放大了数值。
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train
