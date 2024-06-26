# coding=utf-8
import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random


class Corpus:
    def __init__(self, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train, get_2hop=False):
        # train_data, validation_data, test_data��Ԫ�飬����train_data��(train_triples, train_adjacency_mat)
        # train_triples���б�Ԫ����Ԫ��(ͷʵ��id����ϵid��βʵ��id)��
        # train_adjacency_mat��Ԫ��(�ڽӾ�����[ʵ��id...]����[ʵ��id...]���ڽӾ����е�Ԫ��[��ȨΪ1����ȨΪ��ϵid...])
        # entity2id���ֵ�{ʵ�������ַ���: ���id}�� relation2id���ֵ�{��ϵ�����ַ���: ���id}
        # headTailSelector���ֵ�{��ϵid: 1000��ƽ��βʵ����Ŀ/ƽ��ͷβʵ����Ŀ}
        # valid_to_invalid_samples_ratio�ǲ���ȷ����ȷ��Ԫ��ı���
        # unique_entities_train���б�["/m/0b76d_m",...]���б����ַ���Ԫ�ز��ظ�

        self.train_triples = train_data[0]  # train_triples��train_data[0]���б�Ԫ����Ԫ��(ͷʵ��id����ϵid��βʵ��id)

        # Converting to sparse tensor  # train_data[1]��Ԫ�飬(�ڽӾ�����[ʵ��id...]����[ʵ��id...]���ڽӾ����е�Ԫ��[��ȨΪ1����ȨΪ��ϵid...])
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns  # ��ά����(2, ѵ��������Ԫ�����)��Ĭ�ϣ���������(βʵ��id)����������(ͷʵ��id)�����ظ�����Ϊ����ʱ�б���ѵ�����ݵ�ÿһ��
        adj_values = torch.LongTensor(train_data[1][2])  # һά����
        self.train_adj_matrix = (adj_indices, adj_values)  # train_adj_matrix��Ԫ��(��ά����, һά����)

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)  # ����ȷ����ȷ��Ԫ��ı���

        if(get_2hop):  # ���2���ھӽڵ�
            self.graph = self.get_graph()  # ����ͼ��graph�ṹ���ֵ�{ͷʵ��id: {βʵ��id: ��ϵid}}
            self.node_neighbors_2hop = self.get_further_neighbors()
            # node_neighbors_2hop�ھӽڵ��ֵ�{ʵ��id: {����: [((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...))]}}

        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]  # unique_entities_train���б�[ʵ��id,...]

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)  # train_indices��2ά���飬��С(��Ԫ����, 3)��Ԫ�ص�һ����ͷʵ��id,��ϵid,βʵ��id
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)  # train_values��2ά���飬��С(��Ԫ����, 1)��Ԫ�ض���1���൱�����˱�ǩ

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)  # numpy����Ԫ�����ݱ��һ��������
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)  # ������ȷ�ģ����ϱ�ǩ1

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)  # (��Ԫ����, 3)��Ԫ�ص�һ����ͷʵ��id,��ϵid,βʵ��id
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)  # ������ȷ�ģ����ϱ�ǩ1

        self.valid_triples_dict = {j: i for i, j in enumerate(                # +�Ų�����������Ԫ�鶼����һ���ˡ�valid_triples_dict�ֵ�{��Ԫ��(ͷʵ��id, ��ϵid, βʵ��id): ���}������ѵ������֤�Ͳ��Ե�������Ԫ��
            self.train_triples + self.validation_triples + self.test_triples)}  # i�Ǳ��(��0��ʼ)��j����Ԫ��(ͷʵ��id, ��ϵid, βʵ��id)
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(len(self.valid_triples_dict), len(self.train_indices),len(self.validation_indices), len(self.test_indices)))
                                                                                                           # numpy����Ҳ������len()���鿴�ж�����

        # For training purpose
        self.batch_indices = np.empty(                                               # empty�����������ʼ��Ԫ��ֵ
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)  # numpy���飬��С((self.batch_size * (self.invalid_valid_ratio + 1), 3)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

    def get_iteration_batch(self, iter_num):  # ��iter_num�ε���
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):  # �����һ������(iter_num+1)��û������Ԫ����Ŀ
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)  # �����ܵ�(������ȷ�Ͳ���ȷ��Ԫ����Ŀ)һ�����ε�����(��Ԫ��id������)����С(һ��������Ŀ*(����Ԫ���븺��Ԫ����� + 1)��, 3��)
            # ������self.batch_size * (self.invalid_valid_ratio + 1)��������Ϊ2����������� 1 * batch_size����ȷԪ�飬2 * batch_size�в���ȷԪ��

            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)  # ����һ�����ε����ݵı�ǩ

            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))  # һ�����ε��±����������ڴ�ѵ������������

            self.batch_indices[:self.batch_size,
                               :] = self.train_indices[indices, :]  # �����±��������������ݸ�ֵ��ǰbatch_size�У����е���
            self.batch_values[:self.batch_size,
                              :] = self.train_values[indices, :]    # �����±������������������ñ�ǩ

            last_index = self.batch_size  # ��¼���һ���±�����ֵ

            if self.invalid_valid_ratio > 0:  # ѵ��ʱ������ȷ����ȷ��Ԫ��ı��ʣ�������0
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)  # ����ɢ���ȷֲ������ʵ��id������СΪ(last_index * self.invalid_valid_ratio, ) һά���飬Ԫ��Ϊʵ��id
                # ��Ϊ����ȷ����ȷ��Ԫ��ı��ʴ���0����϶�Ҫ����һЩ����ȷ����Ԫ�飬Ȼ��ƴ����������������Ƕ��٣����Ϊ2���������ȷԪ�������Ϊ��ȷԪ������last_index * 2
                # ����Ϊʲôrandom_entities�Ĵ�С��(last_index * self.invalid_valid_ratio, )����Ϊ����ʹ��random_entitiesʱ������������޾���last_index * self.invalid_valid_ratio

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                # ����ȷ��Ԫ���Ǽ�ǰlast_index����Ԫ�飬�ѵ����(last_index * invalid_valid_ratio, 3)��Ȼ�����������ֵ��batch_indices�ĸ���Ԫ���λ��
                # ��ʵ���ǽ�[:last_index, :]�Ķ������Ƶ�[last_index:(last_index*self.invalid_valid_ratio), :]�У�
                # Ȼ���ٽ�[:last_index, :]�Ķ������Ƶ�[(last_index*self.invalid_valid_ratio):(last_index*self.invalid_valid_ratio)+1, :]��

                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))  # �Ա�ǩ����Ҳͬ������

                for i in range(last_index):  # ��ÿһ�����ε���Ԫ��
                    # �����ȶ�ͷʵ�����滻
                    for j in range(self.invalid_valid_ratio // 2):  # j���ڱ������飬�����range(1)�Ļ�����ôֻ�ܲ���0��
                        current_index = i * (self.invalid_valid_ratio // 2) + j  # ��2ȡ������˼������ж��ٸ�����(�ֱ������滻ͷ���滻β)����ʵҲ���Ƕ�ÿ��iѭ������j��
                        # ��Ϊ���������飬����һ�����ڴ���滻ͷʵ��ĸ�����

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],    # ��batch_indices�ĸ���Ԫ����У��������滻��ͷʵ��Ĵ������Ԫ������ȷ��
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():  # �ֵ�{��Ԫ��(ͷʵ��id, ��ϵid, βʵ��id): ���}
                            random_entities[current_index] = np.random.randint(                                       # valid_triples_dict��������������(ѵ�����ݡ���֤���ݡ���������)
                                0, len(self.entity2id))  # ��������滻ͷʵ��(����random_entities[��ǰΪֹ])��ֱ���Ǵ����Ϊֹ
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]  # ȷ���Ǵ������Ԫ���(�滻ͷ)��batch_indices�ĸ����е�ĳһ����Ԫ���������ĸ�����
                        self.batch_values[last_index + current_index, :] = [-1]  # ��Ϊ�Ǵ������Ԫ�飬��˱�ǩ��Ӧ����Ϊ-1

                    # �����ٶ�βʵ�����滻
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * (self.invalid_valid_ratio // 2) + (i * (self.invalid_valid_ratio // 2) + j)
                        # ��Ϊ���������飬����һ�����ڴ���滻βʵ��ĸ����������Ҫ��ǰ���last_index * (self.invalid_valid_ratio // 2)���֣��ж�������������ٿ顣
                        # ���ratio��4�Ļ������������鶼����ͷ���������鶼����β
                        # �����current_index��������ȡ��last_index * self.invalid_valid_ratio�ģ���current_index����Ϊrandom_entities�б�����������random_entities�б�Ĵ�С����Ϊ(last_index * self.invalid_valid_ratio, )

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]  # ȷ���Ǵ������Ԫ���(�滻β)��batch_indices�ĸ����е�ĳһ����Ԫ���������ĸ�����
                        self.batch_values[last_index + current_index, :] = [-1]  # ��Ϊ�Ǵ������Ԫ�飬��˱�ǩ��Ӧ����Ϊ-1

                return self.batch_indices, self.batch_values  # �������ȷ����ȷ��Ԫ��ı���>0���򷵻�

            return self.batch_indices, self.batch_values  # ���򷵻ص�����ֻ�����������޸�����

        else:  # �����һ������(iter_num+1)��������Ԫ����Ŀ����˵��ʣ�µ����ݻ����������������(iter_num)
            last_iter_size = len(self.train_indices) - self.batch_size * iter_num  # �����ʣ�µ�����(����һ������)������

            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)  # ���batch_indices���оͲ���һ��batch_size�ˣ�����last_iter_size
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))  # ��һ�����ε����ݵ�ĩβ�����ڴ�ѵ������������
            self.batch_indices[:last_iter_size,
                               :] = self.train_indices[indices, :]  # �����ݷ���
            self.batch_values[:last_iter_size,
                              :] = self.train_values[indices, :]  # �����ǩ

            last_index = last_iter_size  # ��¼��������������

            if self.invalid_valid_ratio > 0:  # ѵ��ʱ������ȷ����ȷ��Ԫ��ı��ʣ�������0
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)  # ����ɢ���ȷֲ������ʵ��id������СΪ(last_index * self.invalid_valid_ratio, ) һά���飬Ԫ��Ϊʵ��id
                # ����Ϊʲôrandom_entities�Ĵ�С��(last_index * self.invalid_valid_ratio, )����Ϊ����ʹ��random_entitiesʱ������������޾���last_index * self.invalid_valid_ratio

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                # ����ȷ��Ԫ���Ǽ�ǰlast_index����Ԫ�飬�ѵ����(last_index * invalid_valid_ratio, 3)��Ȼ�����������ֵ��batch_indices�ĸ���Ԫ���λ��
                # ��ʵ���ǽ�[:last_index, :]�Ķ������Ƶ�[last_index:(last_index*self.invalid_valid_ratio), :]�У�
                # Ȼ���ٽ�[:last_index, :]�Ķ������Ƶ�[(last_index*self.invalid_valid_ratio):(last_index*self.invalid_valid_ratio)+1, :]��

                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))  # �Ա�ǩҲͬ������

                # ����ѭ���������ѭ��ʱһ����
                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]  # ȷ���Ǵ������Ԫ���(�滻ͷ)��batch_indices�ĸ����е�ĳһ����Ԫ���������ĸ�����
                        self.batch_values[last_index + current_index, :] = [-1]  # ��Ϊ�Ǵ������Ԫ�飬��˱�ǩ��Ӧ����Ϊ-1

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * (self.invalid_valid_ratio // 2) + (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]  # ȷ���Ǵ������Ԫ���(�滻β)��batch_indices�ĸ����е�ĳһ����Ԫ���������ĸ�����
                        self.batch_values[last_index + current_index, :] = [-1]  # ��Ϊ�Ǵ������Ԫ�飬��˱�ǩ��Ӧ����Ϊ-1

                return self.batch_indices, self.batch_values  # �������ȷ����ȷ��Ԫ��ı���>0���򷵻�

            return self.batch_indices, self.batch_values  # ���򷵻ص�����ֻ�����������޸�����

    def get_iteration_batch_nhop(self, current_batch_indices, node_neighbors, batch_size):

        self.batch_indices = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 4)).astype(np.int32)
        self.batch_values = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        indices = random.sample(range(len(current_batch_indices)), batch_size)

        self.batch_indices[:batch_size,
                           :] = current_batch_indices[indices, :]
        self.batch_values[:batch_size,
                          :] = np.ones((batch_size, 1))

        last_index = batch_size

        if self.invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    self.batch_indices[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (self.invalid_valid_ratio // 2) + \
                        (i * (self.invalid_valid_ratio // 2) + j)

                    self.batch_indices[last_index + current_index,
                                       3] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

            return self.batch_indices, self.batch_values

        return self.batch_indices, self.batch_values

    def get_graph(self):
        graph = {}    # graph�ṹ���ֵ�{ͷʵ��id: {βʵ��id: ��ϵid}}
        # train_adj_matrix��Ԫ��(��ά����, һά����)��(adj_indices, adj_values)��train_adj_matrix[0]��һ�����к�(ʵ��id)���ڶ������к�(ʵ��id)
        all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
            0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)  # ��ɵ�һ�����к�(βʵ��id)���ڶ������к�(ͷʵ��id)�����������ڽӾ���Ԫ��ֵ(��ϵ��)

        for data in all_tiples:  # dataȡall_tiples������ÿһ��
            source = data[1].data.item()  # ͷʵ����data��1��
            target = data[0].data.item()  # βʵ����data��0��
            value = data[2].data.item()   # ��ϵ��data��2��

            if(source not in graph.keys()):  # ��ͷʵ�岻��ͼ��
                graph[source] = {}           # �򴴽����ֵ�
                graph[source][target] = value  # ��ͷʵ����ֵ��У�βʵ��idΪ������ϵidΪֵ
            else:
                graph[source][target] = value
        print("Graph created")
        return graph

    def bfs(self, graph, source, nbd_size=2):  # ��ĳ��sourceͷʵ��id�������������  graph�ṹ���ֵ�{ͷʵ��id: {βʵ��id: ��ϵid}}
        visit = {}  # �����ֵ�{ʵ��id: 1} ��1��ʾ�ѷ���
        distance = {}  # �����ֵ�{ʵ��id: ��sourceʵ��ľ���(����)}����ʾʵ����sourceʵ��֮��ľ���
        parent = {}  # ���ڵ��ֵ�{ʵ��id: Ԫ��(���ʵ��ĸ��ڵ�id, ��ϵid)}��(-1, -1)��ʾ������Դͷ
        distance_lengths = {}  # ���볤���ֵ�{�����ڵ���sourceʵ��ľ���: 1}��1��ʾ�Ѿ��нڵ���source�����������

        visit[source] = 1
        distance[source] = 0  # sourceʵ�嵽�Լ��ľ���Ϊ0����·����ÿ��������ʵ�嶼���������Ļ����ϼ�1����ʾ�Լ���sourceʵ��ľ���
        parent[source] = (-1, -1)  # ��ʾsource�Ǳ���bfs��Դͷ�ڵ�

        q = queue.Queue()  # ���д洢Ԫ��Ԫ��(ʵ��id, ��ϵid)
        q.put((source, -1))  # ��Ϊ�Լ����Լ�û�й�ϵ���������ϵidΪ-1

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():  # ʵ����Ϊͼ�е�ͷʵ��
                for target in graph[top[0]].keys():  # ���������ͷʵ������������βʵ��
                    if(target in visit.keys()):  # �����βʵ���Ѿ������ʹ���
                        continue                 # �����������һ��βʵ��
                    else:                                       # ������βʵ�廹û�����ʹ�
                        q.put((target, graph[top[0]][target]))  # ��Ԫ��(βʵ��id, ��ϵid)�ӵ�������

                        distance[target] = distance[top[0]] + 1  # targetʵ�嵽���top[0]ʵ��ľ���

                        visit[target] = 1  # ����Ѿ����ʹ����targetʵ����
                        if distance[target] > 2:  # ���target��sourceʵ��ľ������2
                            continue              # ����Ҫ��¼���£����ڵ���Ƿ��д˾��볤�ȣ�ֻҪ��¼���ϣ��˽ڵ��Ƿ��ѱ����ʣ��˽ڵ���sourceʵ��ľ��룬���������
                        parent[target] = (top[0], graph[top[0]][target])  # (targetʵ��ĸ��ڵ�id, ����֮��Ĺ�ϵid)

                        if distance[target] not in distance_lengths.keys():  # ���target��sourceʵ��ľ��벻��distance_lengths�ֵ���
                            distance_lengths[distance[target]] = 1           # ��distance_lengths�ֵ�ļ��Ǵ˾��룬ֵ��1����һ�±��

        neighbors = {}  # �ھ��ֵ�{����ĳ��ʵ����sourceʵ��ľ���: �б�[((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...))]}�� ����б��Ԫ����Ԫ���Ԫ��
        for target in visit.keys():  # ���ѷ��ʹ���ʵ���У��ó�targetʵ��
            # ֻ���Ǿ���sourceֻ��2������Щʵ��
            if(distance[target] != nbd_size):  # ���targetʵ����sourceʵ��ľ��벻����2���򲻿������target����ȡ��һ��
                continue
            edges = [-1, parent[target][1]]  # ·���б���¼·����[-1, ��ϵid]
            relations = []  # ��ϵ�б�
            entities = [target]  # ʵ���б�[targetʵ��id]���Լ��ȼӽ�ȥ
            temp = target
            while(parent[temp] != (-1, -1)):  # ���tempʵ�廹û��Դͷsource�ڵ㣬��
                relations.append(parent[temp][1])  # ���븸�ڵ�Ĺ�ϵid���뵽��ϵ�б���
                entities.append(parent[temp][0])   # �����ڵ�id���뵽ʵ���б���
                temp = parent[temp][0]  # �൱�ڻ��ݲ�����Ѱ����Դ�������ڵ���Ϊtemp�ڵ�

            if(distance[target] in neighbors.keys()):  # ���targetʵ����sourceʵ��ľ��� ���ھ��ֵ�ļ���
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))  # ����������м���
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]  # �����û����������򴴽��µ��б��б��Ԫ����Ԫ���Ԫ��
                # [((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...))]  targetʵ�嵽sourceʵ���·���ϵ����й�ϵ���ɵ�Ԫ�飬
                #  targetʵ�嵽sourceʵ���·���ϵ�����ʵ�壨���Լ����ڶ���ʵ��Ϊֹ��������ֵ�һ����ʵ�壩���ɵ�Ԫ��

        return neighbors

    def get_further_neighbors(self, nbd_size=2):
        neighbors = {}  # �ھӽڵ��ֵ�{ʵ��id: {����: [((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...))]}}
        # ����ĳ��ʵ�壬֪�������ж��پ�����Ǹ�ʵ���ϵ�·���е�����ʵ��͹�ϵ
        # ������Ļ�����Ϊ��2���ھӣ����Ծ��붼��2
        start_time = time.time()          # graph.keys()����ͷʵ��id��Ŀ��������ֿ��ܱ�ѵ��������ʵ��id�٣�����Ϊֻ������ͷʵ��
        print("length of graph keys is ", len(self.graph.keys()))  # ������Щʵ��ֻ��Ϊβʵ�壬���������ͷ��
        for source in self.graph.keys():  # ����ÿһ��ͷʵ�����
            # st_time = time.time()
            # temp_neighbors���ھ��ֵ�{����ĳ��ʵ����sourceʵ��ľ���: �б�[((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...))]}�� ����б��Ԫ����Ԫ���Ԫ��
            # ȫ���Ǿ���sourceֻ��2����ʵ�壬���temp_neighbors�ֵ�ļ�ȫΪ2
            temp_neighbors = self.bfs(self.graph, source, nbd_size)  # ��ĳ��ͷʵ���������������������ͷʵ�嶼��һ�顣
            for distance in temp_neighbors.keys():  # ����ĳ��ͷʵ����ȥ����ÿһ������
                if(source in neighbors.keys()):  # ��sourceʵ��id�ڼ���
                    if(distance in neighbors[source].keys()):  # ����ĳsourceʵ����ԣ���������Ƿ��ѱ���¼
                        neighbors[source][distance].append(    # ���ѱ���¼����˵����source�ܵ���������룬��ֹ��һ��ʵ��
                            temp_neighbors[distance])          # ���Ҫ������������䣬��ֵ����б��м����б�[Ԫ��((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...))]
                    else:                                      # ע�⣡����ֱ��append�б�
                        neighbors[source][distance] = temp_neighbors[distance]  # û��¼���¼
                else:
                    neighbors[source] = {}  # ���ڼ��У���Ϊsourceʵ��id�������ӿ��ֵ�{}��Ϊֵ
                    neighbors[source][distance] = temp_neighbors[distance]  # �Ѿ�����Ϊ���ֵ�ļ���[((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...))]��Ϊֵ

        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        return neighbors

    def get_batch_nhop_neighbors_all(self, gat_paras, batch_sources, node_neighbors, nbd_size=2):
        # batch_sources���б�[ʵ��id,...]����ѵ������ÿ��ʵ���id�����ظ�
        # node_neighbors�ھӽڵ��ֵ�{ʵ��id: {����: [((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...)),...]}}
        batch_source_triples = []  # �б���б�[[ÿ��sourceʵ��id, ��sourceʵ������Ĺ�ϵid, ��sourceʵ����Զ�Ĺ�ϵid, ��sourceʵ����Զ��ʵ��id],...]
        print("length of unique_entities ", len(batch_sources))
        count = 0
        for source in batch_sources:  # ��ÿһ��ʵ��id�����ظ�
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():  # �������ڵ����ھӽڵ㣬�򽫾���source�ڵ�nbd_size���Ľڵ�͹�ϵ��Ϣ�ó���
                nhop_list = node_neighbors[source][nbd_size]  # nhop_list���б�[((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...)),...]

                for i, tup in enumerate(nhop_list):  # ����ÿһ��((��ϵid, ��ϵid,...), (ʵ��id, ʵ��id,...))Ԫ��
                    if(gat_paras['gat_partial_2hop'] and i >= 1):  # ��partial_2hopΪ�棬��ֻ�õ�һ��2���ھӺ�·������
                        break                          # ��partial_2hopΪ�٣���ȫ���ó����ж����ö���

                    count += 1  # �б�[sourceʵ��id, ��sourceʵ������Ĺ�ϵid, ��sourceʵ����Զ�Ĺ�ϵid, ��sourceʵ����Զ��ʵ��id]
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                                 nhop_list[i][1][0]])  # ���б���뵽������

        return np.array(batch_source_triples).astype(np.int32)
        # args.partial_2hopΪ�棺����numpy���飬��С(ʵ����Ŀ, 4)�����ظ���4��ʾÿ��ʵ���id, ��sourceʵ������Ĺ�ϵid, ��sourceʵ����Զ�Ĺ�ϵid����sourceʵ����Զ��ʵ��id
        # args.partial_2hopΪ�٣�

    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x

    def get_validation_pred(self,gat_data_path, model, unique_entities):
        # ���ģ��model
        # unique_entities���б�[ʵ��id,...]����ѵ�������в��ظ���ʵ���id

        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):  # range(1)ֵ[0, 1)��ִֻ��һ�Σ���
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]  # ���ݲ���������Ԫ���������������б�[0,1,2...]
            batch_indices = self.test_indices[indices, :]  # batch_indices��(����������Ԫ����, 3)����һ��Ԫ��[[ʵ��id, ��ϵid, ʵ��id],...]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))  # ���еĲ�������
            entity_list = [j for i, j in self.entity2id.items()]  # entity2id���ֵ�{ʵ�������ַ���: ���id}��entity_list���б�[ʵ��id, ʵ��id,...]����������ʵ��ı�ţ���ѵ�������Ժ���֤�����ж����ֵ�

            ranks_head, ranks_tail = [], []  # �洢������ȷ�Ĳ�����Ԫ���ͷʵ���βʵ�������
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_indices.shape[0]):  # ����ÿһ����ȷ�ġ����Ե���Ԫ��
                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))  # ����i��������Ԫ��������¶ѵ����γ�(ʵ���ܸ���, 3)������һ����[ʵ��id,��ϵid,ʵ��id]���ڶ���Ҳ��[ʵ��id,��ϵid,ʵ��id]
                new_x_batch_tail = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))  # ����i��������Ԫ��������¶ѵ����γ�(ʵ���ܸ���, 3)������һ����[ʵ��id,��ϵid,ʵ��id]���ڶ���Ҳ��[ʵ��id,��ϵid,ʵ��id]

                if batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities:
                    continue  # ���(��i��������Ԫ���ͷʵ�� û����ѵ�������г���) ���� (��i��������Ԫ���βʵ�� û����ѵ�������г���)�����������������Ԫ�飬��������Ԫ�飬��������

                new_x_batch_head[:, 0] = entity_list  # ������ʵ���id ����ֵ�� new_x_batch_head��ͷʵ����
                new_x_batch_tail[:, 2] = entity_list  # ������ʵ���id ����ֵ�� new_x_batch_tail��βʵ����

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):  # ��һ��ʵ��(��ʵ����������ʼ����)
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])  # �ó�ÿһ���滻��ͷ�˵���Ԫ�飬
                    if temp_triple_head in self.valid_triples_dict.keys():  # valid_triples_dict�ֵ�{��Ԫ��(ͷʵ��id, ��ϵid, βʵ��id): ���}������ѵ������֤�Ͳ��Ե�������Ԫ��
                        # ������������һ����Ԫ��������������ݼ��У�
                        last_index_head.append(tmp_index)  # ��tmp_index��new_x_batch_head�ĵڼ��У����ڼ���ͷʵ�����ڵ��Ǹ���Ԫ��������ţ���¼���б���
                        # ע�⣡���������������õ��Ǹ���Ԫ��ض���������������ݼ��У����������ڲ��Ե���Ԫ��ض��ᱻɾ������˺������н������·�����˵Ĳ���

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])  # �ó�ÿһ���滻��β�˵���Ԫ�飬
                    if temp_triple_tail in self.valid_triples_dict.keys():  # ������������һ����Ԫ��������������ݼ��У�
                        last_index_tail.append(tmp_index)  # ��tmp_index��new_x_batch_head�ĵڼ��У����ڼ���βʵ�����ڵ��Ǹ���Ԫ��������ţ���¼���б���

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them may not be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)  # ��0ά������last_index_head�е������ţ���new_x_batch_head��ɾ��һ����Ԫ������
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)  # ��0ά������last_index_tail�е������ţ���new_x_batch_tail��ɾ��һ����Ԫ������

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)  # ��new_x_batch_head�е���ˣ�������ȷ�ġ����Ե���Ԫ��
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)  # ��new_x_batch_tail�е���ˣ�������ȷ�ġ����Ե���Ԫ��

                # ��new_x_batch_headΪ���ӣ����Ƕ�άnumpy����(δ֪����, 3)����һ�������ڲ��Ե���Ԫ��[[ʵ��1id, ��ϵ1id, ʵ��2id],...
                # �ڶ����������滻��ͷʵ�����Ԫ��[ʵ��2id, ��ϵ1id, ʵ��2id],...]

                import math
                # Have to do this, because it doesn't fit in memory

                if 'wn' in gat_data_path:  # WN18RR���ݼ��Ļ�
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_head.shape[0] / 10))  # ��new_x_batch_head�ֳ�4��

                    scores1_head = model.batch_test(torch.from_numpy(
                        new_x_batch_head[:num_triples_each_shot, :]).long().cuda())
                    # ��new_x_batch_head��һ����Ϊ���룬�þ��ģ��������
                    # ���صĽ��scores1_head������(num_triples_each_shot, 1)���Ǿ�����������logitֵ����ʾ��������ϳ���ÿһ����Ԫ��logitֵ

                    scores2_head = model.batch_test(torch.from_numpy(
                        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).long().cuda())
                    scores3_head = model.batch_test(torch.from_numpy(
                        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).long().cuda())
                    scores4_head = model.batch_test(torch.from_numpy(
                        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).long().cuda())
                    scores5_head = model.batch_test(torch.from_numpy(
                         new_x_batch_head[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).long().cuda())
                    scores6_head = model.batch_test(torch.from_numpy(
                         new_x_batch_head[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).long().cuda())
                    scores7_head = model.batch_test(torch.from_numpy(
                         new_x_batch_head[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).long().cuda())
                    scores8_head = model.batch_test(torch.from_numpy(
                         new_x_batch_head[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).long().cuda())
                    scores9_head = model.batch_test(torch.from_numpy(
                         new_x_batch_head[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).long().cuda())
                    scores10_head = model.batch_test(torch.from_numpy(
                         new_x_batch_head[9 * num_triples_each_shot:, :]).long().cuda())

                    scores_head = torch.cat(
                        [scores1_head, scores2_head, scores3_head, scores4_head,
                        scores5_head, scores6_head, scores7_head, scores8_head,
                        scores9_head, scores10_head], dim=0)
                else:  # ������WN18RR���ݼ�����ȫ���Ž�ȥ
                    scores_head = model.batch_test(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)  # ��scores_headչƽ֮�󣬰�������������
                # sorted_scores_head������(new_x_batch_head.shape[0], )�����ź���ķ���
                # sorted_indices_head������(new_x_batch_head.shape[0], )�����ź���ķ�����ԭ��scores_head.view(-1)�е�λ��

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().data.numpy() == 0)[0][0] + 1)  # where�ҳ�ԭ����������Ϊ0����ȷ��Ԫ���λ�ã�+1����Ϊ�����Ǵ�0��ʼ�ģ��������Ǵ�1��ʼ��
                # �ҵ���ȷ��Ԫ�������֮�󣬽����������뵽ranks_head�б��С�[��һ����Ԫ�������, �ڶ���,...]
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])  # ���ոռ���������ĵ������뵽reciprocal_ranks_head�б���
                # ��ô���ˣ�һ����ȷ��Ԫ���ͷʵ���������Ծͽ�����

                # Tail part here  βʵ��Ҳ��һ����

                if 'wn' in gat_data_path:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_tail.shape[0] / 10))

                    scores1_tail = model.batch_test(torch.from_numpy(
                        new_x_batch_tail[:num_triples_each_shot, :]).long().cuda())
                    scores2_tail = model.batch_test(torch.from_numpy(
                        new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).long().cuda())
                    scores3_tail = model.batch_test(torch.from_numpy(
                        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).long().cuda())
                    scores4_tail = model.batch_test(torch.from_numpy(
                        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).long().cuda())
                    scores5_tail = model.batch_test(torch.from_numpy(
                         new_x_batch_tail[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).long().cuda())
                    scores6_tail = model.batch_test(torch.from_numpy(
                         new_x_batch_tail[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).long().cuda())
                    scores7_tail = model.batch_test(torch.from_numpy(
                         new_x_batch_tail[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).long().cuda())
                    scores8_tail = model.batch_test(torch.from_numpy(
                         new_x_batch_tail[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).long().cuda())
                    scores9_tail = model.batch_test(torch.from_numpy(
                         new_x_batch_tail[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).long().cuda())
                    scores10_tail = model.batch_test(torch.from_numpy(
                         new_x_batch_tail[9 * num_triples_each_shot:, :]).long().cuda())

                    scores_tail = torch.cat(
                        [scores1_tail, scores2_tail, scores3_tail, scores4_tail,# dim=0)
                         scores5_tail, scores6_tail, scores7_tail, scores8_tail,
                         scores9_tail, scores10_tail], dim=0)

                else:
                    scores_tail = model.batch_test(new_x_batch_tail)

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().data.numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])

                print("sample - ", ranks_head[-1], ranks_tail[-1])  # ������������Թ۲�һ������Ľ�������Կ�����һЩ����̫�����ˣ����Բ鿴ԭ��

            for i in range(len(ranks_head)):  # ��ͷʵ�����
                if ranks_head[i] <= 100:   # ������С��100�����¼���ۼ���Ŀ
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:    # ������С��10�����¼���ۼ���Ŀ
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:    # ������С��3�����¼���ۼ���Ŀ
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:    # ������С��1�����¼���ۼ���Ŀ
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)  # ��֤ ͷʵ����������б��ͷʵ����������������б� Ԫ�������Ƿ�һ��
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("here {}".format(len(ranks_head)))  # WN18RR��������ֲ������ڲ�����Ԫ�����Ŀ����Ϊ��������ȥ����һЩ��Ԫ��(��Щ��Ԫ���е�ͷʵ���βʵ����ѵ��������û�г��ֹ�)
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")  # ͷʵ������
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))  # ����������ȡƽ��

            print("\nStats for replacing tail are -> ")  # βʵ������
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_tail / len(ranks_head)))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_tail / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_tail / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_tail / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_tail) / len(ranks_tail)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))  # ͷʵ������С��100�ĸ��� �� ������ʵ��ĸ���������ͷʵ������С��100�ı���
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))  # ����ͷʵ������������ �� ������ʵ��ĸ���������ƽ��ͷʵ������
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))    # ����ƽ��ͷʵ���MRR

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        print("\nAveraged stats for replacing head are -> ")  # ��Ϊfor iters in range(1):  ִֻ��һ�Σ����������������Ľ���������һ��
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

        print("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2  # ͷƽ��hit@100 ���� βƽ��hit@100���ٳ�2��������ͷ��β��ƽ��hit@100
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2  # ͷƽ��MR ���� βƽ��MR���ٳ�2��������ͷ��β��ƽ��MR
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2  # ͷƽ��MRR ���� βƽ��MRR���ٳ�2��������ͷ��β��ƽ��MRR

        print("\nCumulative stats are -> ")   # ����������
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))

