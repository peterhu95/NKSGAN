# coding=utf-8
import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random


class Corpus:
    def __init__(self, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train, get_2hop=False):
        # train_data, validation_data, test_data是元组，例如train_data是(train_triples, train_adjacency_mat)
        # train_triples是列表，元素是元组(头实体id，关系id，尾实体id)，
        # train_adjacency_mat是元组(邻接矩阵行[实体id...]，列[实体id...]，邻接矩阵中的元素[无权为1，有权为关系id...])
        # entity2id是字典{实体名称字符串: 编号id}， relation2id是字典{关系名称字符串: 编号id}
        # headTailSelector是字典{关系id: 1000乘平均尾实体数目/平均头尾实体数目}
        # valid_to_invalid_samples_ratio是不正确与正确三元组的比例
        # unique_entities_train是列表["/m/0b76d_m",...]，列表中字符串元素不重复

        self.train_triples = train_data[0]  # train_triples是train_data[0]是列表，元素是元组(头实体id，关系id，尾实体id)

        # Converting to sparse tensor  # train_data[1]是元组，(邻接矩阵行[实体id...]，列[实体id...]，邻接矩阵中的元素[无权为1，有权为关系id...])
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns  # 二维张量(2, 训练数据三元组个数)。默认：上面是行(尾实体id)，下面是列(头实体id)，有重复，因为构建时有遍历训练数据的每一行
        adj_values = torch.LongTensor(train_data[1][2])  # 一维张量
        self.train_adj_matrix = (adj_indices, adj_values)  # train_adj_matrix是元组(二维张量, 一维张量)

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
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)  # 不正确与正确三元组的比例

        if(get_2hop):  # 获得2跳邻居节点
            self.graph = self.get_graph()  # 构造图，graph结构：字典{头实体id: {尾实体id: 关系id}}
            self.node_neighbors_2hop = self.get_further_neighbors()
            # node_neighbors_2hop邻居节点字典{实体id: {距离: [((关系id, 关系id,...), (实体id, 实体id,...))]}}

        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]  # unique_entities_train是列表[实体id,...]

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)  # train_indices是2维数组，大小(三元组数, 3)，元素第一行是头实体id,关系id,尾实体id
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)  # train_values是2维数组，大小(三元组数, 1)，元素都是1。相当于做了标签

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)  # numpy将三元组数据变成一行数据了
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)  # 都是正确的，打上标签1

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)  # (三元组数, 3)，元素第一行是头实体id,关系id,尾实体id
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)  # 都是正确的，打上标签1

        self.valid_triples_dict = {j: i for i, j in enumerate(                # +号操作将所有三元组都放在一起了。valid_triples_dict字典{三元组(头实体id, 关系id, 尾实体id): 编号}，包含训练、验证和测试的所有三元组
            self.train_triples + self.validation_triples + self.test_triples)}  # i是编号(从0开始)，j是三元组(头实体id, 关系id, 尾实体id)
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(len(self.valid_triples_dict), len(self.train_indices),len(self.validation_indices), len(self.test_indices)))
                                                                                                           # numpy数组也可以用len()，查看有多少项

        # For training purpose
        self.batch_indices = np.empty(                                               # empty函数，随机初始化元素值
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)  # numpy数组，大小((self.batch_size * (self.invalid_valid_ratio + 1), 3)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

    def get_iteration_batch(self, iter_num):  # 第iter_num次迭代
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):  # 如果下一个批次(iter_num+1)还没超过三元组数目
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)  # 定义总的(包含正确和不正确三元组数目)一个批次的数据(三元组id号索引)，大小(一个批次数目*(正三元组与负三元组比率 + 1)行, 3列)
            # 行数：self.batch_size * (self.invalid_valid_ratio + 1)。若比率为2，则这里包括 1 * batch_size行正确元组，2 * batch_size行不正确元组

            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)  # 定义一个批次的数据的标签

            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))  # 一个批次的下标索引，用于从训练数据中索引

            self.batch_indices[:self.batch_size,
                               :] = self.train_indices[indices, :]  # 根据下标索引对批次数据赋值，前batch_size行，所有的列
            self.batch_values[:self.batch_size,
                              :] = self.train_values[indices, :]    # 根据下标索引对批次数据做好标签

            last_index = self.batch_size  # 记录最后一个下标索引值

            if self.invalid_valid_ratio > 0:  # 训练时，不正确与正确三元组的比率，若大于0
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)  # 按离散均匀分布随机按实体id产生大小为(last_index * self.invalid_valid_ratio, ) 一维数组，元素为实体id
                # 因为不正确与正确三元组的比率大于0，则肯定要生成一些不正确的三元组，然后拼起来。看这个比率是多少，如果为2，则最后不正确元组的数量为正确元组数量last_index * 2
                # 这里为什么random_entities的大小是(last_index * self.invalid_valid_ratio, )，因为后面使用random_entities时的索引最大上限就是last_index * self.invalid_valid_ratio

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                # 将正确三元组们即前last_index个三元组，堆叠变成(last_index * invalid_valid_ratio, 3)，然后将这个东西赋值到batch_indices的负三元组的位置
                # 其实就是将[:last_index, :]的东西复制到[last_index:(last_index*self.invalid_valid_ratio), :]中，
                # 然后再将[:last_index, :]的东西复制到[(last_index*self.invalid_valid_ratio):(last_index*self.invalid_valid_ratio)+1, :]中

                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))  # 对标签数组也同样操作

                for i in range(last_index):  # 对每一个批次的三元组
                    # 以下先对头实体做替换
                    for j in range(self.invalid_valid_ratio // 2):  # j用于遍历负块，如果是range(1)的话，那么只能产生0了
                        current_index = i * (self.invalid_valid_ratio // 2) + j  # 除2取整的意思是算出有多少个两组(分别用于替换头和替换尾)。其实也就是对每次i循环，做j次
                        # 因为负块有两块，上面一块用于存放替换头实体的负样本

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],    # 在batch_indices的负三元组块中，如果这个替换了头实体的错误的三元组是正确的
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():  # 字典{三元组(头实体id, 关系id, 尾实体id): 编号}
                            random_entities[current_index] = np.random.randint(                                       # valid_triples_dict包含了所有数据(训练数据、验证数据、测试数据)
                                0, len(self.entity2id))  # 则再随机替换头实体(更改random_entities[当前为止])，直到是错误的为止
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]  # 确认是错误的三元组后，(替换头)将batch_indices的负块中的某一个三元组变成真正的负样本
                        self.batch_values[last_index + current_index, :] = [-1]  # 因为是错误的三元组，因此标签相应的设为-1

                    # 以下再对尾实体做替换
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * (self.invalid_valid_ratio // 2) + (i * (self.invalid_valid_ratio // 2) + j)
                        # 因为负块有两块，下面一块用于存放替换尾实体的负样本，因此要有前面的last_index * (self.invalid_valid_ratio // 2)部分，有多少组就跳过多少块。
                        # 如果ratio是4的话，则上面两块都是做头，下面两块都是做尾
                        # 这里的current_index的上限是取到last_index * self.invalid_valid_ratio的，而current_index又作为random_entities列表的索引，因此random_entities列表的大小必须为(last_index * self.invalid_valid_ratio, )

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]  # 确认是错误的三元组后，(替换尾)将batch_indices的负块中的某一个三元组变成真正的负样本
                        self.batch_values[last_index + current_index, :] = [-1]  # 因为是错误的三元组，因此标签相应的设为-1

                return self.batch_indices, self.batch_values  # 如果不正确与正确三元组的比率>0，则返回

            return self.batch_indices, self.batch_values  # 否则返回的数组只有正样本，无负样本

        else:  # 如果下一个批次(iter_num+1)超过了三元组数目，则说明剩下的数据还不够满足这个批次(iter_num)
            last_iter_size = len(self.train_indices) - self.batch_size * iter_num  # 则计算剩下的数据(不够一个批次)的数量

            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)  # 因此batch_indices的行就不满一个batch_size了，而是last_iter_size
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))  # 上一个批次到数据的末尾，用于从训练数据中索引
            self.batch_indices[:last_iter_size,
                               :] = self.train_indices[indices, :]  # 将数据放入
            self.batch_values[:last_iter_size,
                              :] = self.train_values[indices, :]  # 放入标签

            last_index = last_iter_size  # 记录正样本最后的索引

            if self.invalid_valid_ratio > 0:  # 训练时，不正确与正确三元组的比率，若大于0
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)  # 按离散均匀分布随机按实体id产生大小为(last_index * self.invalid_valid_ratio, ) 一维数组，元素为实体id
                # 这里为什么random_entities的大小是(last_index * self.invalid_valid_ratio, )，因为后面使用random_entities时的索引最大上限就是last_index * self.invalid_valid_ratio

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                # 将正确三元组们即前last_index个三元组，堆叠变成(last_index * invalid_valid_ratio, 3)，然后将这个东西赋值到batch_indices的负三元组的位置
                # 其实就是将[:last_index, :]的东西复制到[last_index:(last_index*self.invalid_valid_ratio), :]中，
                # 然后再将[:last_index, :]的东西复制到[(last_index*self.invalid_valid_ratio):(last_index*self.invalid_valid_ratio)+1, :]中

                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))  # 对标签也同样操作

                # 以下循环跟上面的循环时一样的
                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]  # 确认是错误的三元组后，(替换头)将batch_indices的负块中的某一个三元组变成真正的负样本
                        self.batch_values[last_index + current_index, :] = [-1]  # 因为是错误的三元组，因此标签相应的设为-1

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * (self.invalid_valid_ratio // 2) + (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]  # 确认是错误的三元组后，(替换尾)将batch_indices的负块中的某一个三元组变成真正的负样本
                        self.batch_values[last_index + current_index, :] = [-1]  # 因为是错误的三元组，因此标签相应的设为-1

                return self.batch_indices, self.batch_values  # 如果不正确与正确三元组的比率>0，则返回

            return self.batch_indices, self.batch_values  # 否则返回的数组只有正样本，无负样本

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
        graph = {}    # graph结构：字典{头实体id: {尾实体id: 关系id}}
        # train_adj_matrix是元组(二维张量, 一维张量)即(adj_indices, adj_values)。train_adj_matrix[0]第一行是行号(实体id)，第二行是列号(实体id)
        all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
            0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)  # 变成第一列是行号(尾实体id)，第二列是列号(头实体id)，第三列是邻接矩阵元素值(关系号)

        for data in all_tiples:  # data取all_tiples张量的每一行
            source = data[1].data.item()  # 头实体是data的1列
            target = data[0].data.item()  # 尾实体是data的0列
            value = data[2].data.item()   # 关系是data的2列

            if(source not in graph.keys()):  # 若头实体不在图中
                graph[source] = {}           # 则创建空字典
                graph[source][target] = value  # 在头实体的字典中，尾实体id为键，关系id为值
            else:
                graph[source][target] = value
        print("Graph created")
        return graph

    def bfs(self, graph, source, nbd_size=2):  # 对某个source头实体id做宽度优先搜索  graph结构：字典{头实体id: {尾实体id: 关系id}}
        visit = {}  # 访问字典{实体id: 1} ，1表示已访问
        distance = {}  # 距离字典{实体id: 到source实体的距离(跳数)}，表示实体离source实体之间的距离
        parent = {}  # 父节点字典{实体id: 元组(这个实体的父节点id, 关系id)}，(-1, -1)表示真正的源头
        distance_lengths = {}  # 距离长度字典{其他节点离source实体的距离: 1}，1表示已经有节点离source是这个距离了

        visit[source] = 1
        distance[source] = 0  # source实体到自己的距离为0，在路径上每遇到其他实体都在这个距离的基础上加1，表示自己离source实体的距离
        parent[source] = (-1, -1)  # 表示source是本次bfs的源头节点

        q = queue.Queue()  # 队列存储元组元素(实体id, 关系id)
        q.put((source, -1))  # 因为自己与自己没有关系相连，则关系id为-1

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():  # 实体作为图中的头实体
                for target in graph[top[0]].keys():  # 对于与这个头实体相连的所有尾实体
                    if(target in visit.keys()):  # 如果此尾实体已经被访问过了
                        continue                 # 则继续访问下一个尾实体
                    else:                                       # 如果这个尾实体还没被访问过
                        q.put((target, graph[top[0]][target]))  # 则将元组(尾实体id, 关系id)加到队列中

                        distance[target] = distance[top[0]] + 1  # target实体到这个top[0]实体的距离

                        visit[target] = 1  # 标记已经访问过这个target实体了
                        if distance[target] > 2:  # 如果target离source实体的距离大于2
                            continue              # 则不需要记录（下）父节点和是否有此距离长度，只要记录（上）此节点是否已被访问，此节点离source实体的距离，并加入队列
                        parent[target] = (top[0], graph[top[0]][target])  # (target实体的父节点id, 它们之间的关系id)

                        if distance[target] not in distance_lengths.keys():  # 如果target离source实体的距离不在distance_lengths字典中
                            distance_lengths[distance[target]] = 1           # 则distance_lengths字典的键是此距离，值是1，做一下标记

        neighbors = {}  # 邻居字典{其他某个实体离source实体的距离: 列表[((关系id, 关系id,...), (实体id, 实体id,...))]}， 这个列表的元素是元组的元组
        for target in visit.keys():  # 从已访问过的实体中，拿出target实体
            # 只考虑距离source只有2跳的那些实体
            if(distance[target] != nbd_size):  # 如果target实体离source实体的距离不等于2，则不考虑这个target，再取下一个
                continue
            edges = [-1, parent[target][1]]  # 路径列表，记录路径，[-1, 关系id]
            relations = []  # 关系列表
            entities = [target]  # 实体列表[target实体id]，自己先加进去
            temp = target
            while(parent[temp] != (-1, -1)):  # 如果temp实体还没到源头source节点，则
                relations.append(parent[temp][1])  # 将与父节点的关系id加入到关系列表中
                entities.append(parent[temp][0])   # 将父节点id加入到实体列表中
                temp = parent[temp][0]  # 相当于回溯操作（寻根溯源），父节点作为temp节点

            if(distance[target] in neighbors.keys()):  # 如果target实体离source实体的距离 在邻居字典的键中
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))  # 则在这个键中加入
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]  # 如果还没有这个键，则创建新的列表，列表的元素是元组的元组
                # [((关系id, 关系id,...), (实体id, 实体id,...))]  target实体到source实体的路径上的所有关系构成的元组，
                #  target实体到source实体的路径上的所有实体（从自己到第二跳实体为止，不会出现第一跳的实体）构成的元组

        return neighbors

    def get_further_neighbors(self, nbd_size=2):
        neighbors = {}  # 邻居节点字典{实体id: {距离: [((关系id, 关系id,...), (实体id, 实体id,...))]}}
        # 表明某个实体，知道跟我有多少距离的那个实体上的路径中的所有实体和关系
        # 但这里的话，因为是2跳邻居，所以距离都是2
        start_time = time.time()          # graph.keys()键是头实体id数目，这个数字可能比训练数据中实体id少，是因为只考虑了头实体
        print("length of graph keys is ", len(self.graph.keys()))  # 可能有些实体只作为尾实体，不会出现在头中
        for source in self.graph.keys():  # 对于每一个头实体而言
            # st_time = time.time()
            # temp_neighbors是邻居字典{其他某个实体离source实体的距离: 列表[((关系id, 关系id,...), (实体id, 实体id,...))]}， 这个列表的元素是元组的元组
            # 全都是距离source只有2跳的实体，因此temp_neighbors字典的键全为2
            temp_neighbors = self.bfs(self.graph, source, nbd_size)  # 对某个头实体做宽度优先搜索，所有头实体都做一遍。
            for distance in temp_neighbors.keys():  # 对于某个头实体能去到的每一个距离
                if(source in neighbors.keys()):  # 若source实体id在键中
                    if(distance in neighbors[source].keys()):  # 看对某source实体而言，这个距离是否已被记录
                        neighbors[source][distance].append(    # 若已被记录，则说明从source能到的这个距离，不止有一个实体
                            temp_neighbors[distance])          # 因此要距离这个键不变，在值这个列表中加入列表[元组((关系id, 关系id,...), (实体id, 实体id,...))]
                    else:                                      # 注意！这里直接append列表
                        neighbors[source][distance] = temp_neighbors[distance]  # 没记录则记录
                else:
                    neighbors[source] = {}  # 不在键中，则为source实体id键，增加空字典{}作为值
                    neighbors[source][distance] = temp_neighbors[distance]  # 把距离作为新字典的键，[((关系id, 关系id,...), (实体id, 实体id,...))]作为值

        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        return neighbors

    def get_batch_nhop_neighbors_all(self, gat_paras, batch_sources, node_neighbors, nbd_size=2):
        # batch_sources是列表[实体id,...]，是训练数据每个实体的id，不重复
        # node_neighbors邻居节点字典{实体id: {距离: [((关系id, 关系id,...), (实体id, 实体id,...)),...]}}
        batch_source_triples = []  # 列表的列表[[每个source实体id, 离source实体最近的关系id, 离source实体最远的关系id, 离source实体最远的实体id],...]
        print("length of unique_entities ", len(batch_sources))
        count = 0
        for source in batch_sources:  # 对每一个实体id，不重复
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():  # 如果这个节点有邻居节点，则将距离source节点nbd_size跳的节点和关系信息拿出来
                nhop_list = node_neighbors[source][nbd_size]  # nhop_list是列表[((关系id, 关系id,...), (实体id, 实体id,...)),...]

                for i, tup in enumerate(nhop_list):  # 对于每一对((关系id, 关系id,...), (实体id, 实体id,...))元组
                    if(gat_paras['gat_partial_2hop'] and i >= 1):  # 若partial_2hop为真，则只拿第一个2跳邻居和路径出来
                        break                          # 若partial_2hop为假，则全部拿出，有多少拿多少

                    count += 1  # 列表[source实体id, 离source实体最近的关系id, 离source实体最远的关系id, 离source实体最远的实体id]
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                                 nhop_list[i][1][0]])  # 将列表加入到批次中

        return np.array(batch_source_triples).astype(np.int32)
        # args.partial_2hop为真：返回numpy数组，大小(实体数目, 4)，不重复，4表示每个实体的id, 离source实体最近的关系id, 离source实体最远的关系id、离source实体最远的实体id
        # args.partial_2hop为假：

    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x

    def get_validation_pred(self,gat_data_path, model, unique_entities):
        # 卷积模型model
        # unique_entities是列表[实体id,...]，是训练数据中不重复的实体的id

        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):  # range(1)值[0, 1)，只执行一次！！
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]  # 根据测试数据三元组数，生成索引列表，[0,1,2...]
            batch_indices = self.test_indices[indices, :]  # batch_indices是(测试数据三元组数, 3)，第一行元素[[实体id, 关系id, 实体id],...]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))  # 所有的测试数据
            entity_list = [j for i, j in self.entity2id.items()]  # entity2id是字典{实体名称字符串: 编号id}，entity_list是列表[实体id, 实体id,...]，包含所有实体的编号，在训练、测试和验证数据中都出现的

            ranks_head, ranks_tail = [], []  # 存储所有正确的测试三元组的头实体和尾实体的排名
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_indices.shape[0]):  # 对于每一个正确的、测试的三元组
                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))  # 将第i个测试三元组从上往下堆叠，形成(实体总个数, 3)，即第一行是[实体id,关系id,实体id]，第二行也是[实体id,关系id,实体id]
                new_x_batch_tail = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))  # 将第i个测试三元组从上往下堆叠，形成(实体总个数, 3)，即第一行是[实体id,关系id,实体id]，第二行也是[实体id,关系id,实体id]

                if batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities:
                    continue  # 如果(第i个测试三元组的头实体 没有在训练数据中出现) 或者 (第i个测试三元组的尾实体 没有在训练数据中出现)，则舍弃这个测试三元组，跳过此三元组，不做测试

                new_x_batch_head[:, 0] = entity_list  # 将所有实体的id 都赋值到 new_x_batch_head的头实体中
                new_x_batch_tail[:, 2] = entity_list  # 将所有实体的id 都赋值到 new_x_batch_tail的尾实体中

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):  # 对一个实体(从实体总数量开始遍历)
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])  # 拿出每一个替换了头了的三元组，
                    if temp_triple_head in self.valid_triples_dict.keys():  # valid_triples_dict字典{三元组(头实体id, 关系id, 尾实体id): 编号}，包含训练、验证和测试的所有三元组
                        # 如果构造的这样一个三元组出现在整个数据集中，
                        last_index_head.append(tmp_index)  # 则将tmp_index即new_x_batch_head的第几行，即第几个头实体所在的那个三元组的索引号，记录在列表中
                        # 注意！！这里我们正在用的那个三元组必定会出现在整个数据集中，因此这个正在测试的三元组必定会被删除，因此后来才有将它重新放在最顶端的操作

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])  # 拿出每一个替换了尾了的三元组，
                    if temp_triple_tail in self.valid_triples_dict.keys():  # 如果构造的这样一个三元组出现在整个数据集中，
                        last_index_tail.append(tmp_index)  # 则将tmp_index即new_x_batch_head的第几行，即第几个尾实体所在的那个三元组的索引号，记录在列表中

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them may not be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)  # 按0维，根据last_index_head中的索引号，从new_x_batch_head中删除一条三元组数据
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)  # 按0维，根据last_index_tail中的索引号，从new_x_batch_tail中删除一条三元组数据

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)  # 在new_x_batch_head中的最顶端，插入正确的、测试的三元组
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)  # 在new_x_batch_tail中的最顶端，插入正确的、测试的三元组

                # 以new_x_batch_head为例子，它是二维numpy数组(未知数量, 3)，第一行是正在测试的三元组[[实体1id, 关系1id, 实体2id],...
                # 第二行是其他替换了头实体的三元组[实体2id, 关系1id, 实体2id],...]

                import math
                # Have to do this, because it doesn't fit in memory

                if 'wn' in gat_data_path:  # WN18RR数据集的话
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_head.shape[0] / 10))  # 将new_x_batch_head分成4份

                    scores1_head = model.batch_test(torch.from_numpy(
                        new_x_batch_head[:num_triples_each_shot, :]).long().cuda())
                    # 将new_x_batch_head第一份作为输入，用卷积模型做测试
                    # 返回的结果scores1_head是张量(num_triples_each_shot, 1)。是卷积网络输出的logit值。表示神经网络拟合出的每一个三元组logit值

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
                else:  # 若不是WN18RR数据集，则全部放进去
                    scores_head = model.batch_test(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)  # 将scores_head展平之后，按分数降序排列
                # sorted_scores_head是张量(new_x_batch_head.shape[0], )，是排好序的分数
                # sorted_indices_head是张量(new_x_batch_head.shape[0], )，是排好序的分数在原来scores_head.view(-1)中的位置

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().data.numpy() == 0)[0][0] + 1)  # where找出原本分数索引为0即正确三元组的位置，+1是因为索引是从0开始的，但排名是从1开始的
                # 找到正确三元组的排名之后，将此排名加入到ranks_head列表中。[第一个三元组的排名, 第二个,...]
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])  # 将刚刚加入的排名的倒数加入到reciprocal_ranks_head列表中
                # 那么至此，一个正确三元组的头实体排名测试就结束了

                # Tail part here  尾实体也是一样的

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

                print("sample - ", ranks_head[-1], ranks_tail[-1])  # 输出排名。可以观察一下输出的结果，可以看到有一些排名太靠后了，可以查看原因

            for i in range(len(ranks_head)):  # 对头实体而言
                if ranks_head[i] <= 100:   # 若排名小于100，则记录并累计数目
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:    # 若排名小于10，则记录并累计数目
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:    # 若排名小于3，则记录并累计数目
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:    # 若排名小于1，则记录并累计数目
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

            assert len(ranks_head) == len(reciprocal_ranks_head)  # 验证 头实体的排名的列表和头实体计算排名倒数的列表 元素数量是否一致
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("here {}".format(len(ranks_head)))  # WN18RR：这个数字并不等于测试三元组的数目，因为作者这里去除了一些三元组(这些三元组中的头实体或尾实体在训练数据中没有出现过)
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")  # 头实体的情况
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
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))  # 除数量，即取平均

            print("\nStats for replacing tail are -> ")  # 尾实体的情况
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
                hits_at_100_head / len(ranks_head))  # 头实体排名小于100的个数 除 测试总实体的个数。计算头实体排名小于100的比率
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))  # 所有头实体排名加起来 除 测试总实体的个数。计算平均头实体排名
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))    # 计算平均头实体的MRR

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

        print("\nAveraged stats for replacing head are -> ")  # 因为for iters in range(1):  只执行一次！！，因此这里输出的结果和上面的一致
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
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2  # 头平均hit@100 加上 尾平均hit@100，再除2。即计算头和尾的平均hit@100
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2  # 头平均MR 加上 尾平均MR，再除2。即计算头和尾的平均MR
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2  # 头平均MRR 加上 尾平均MRR，再除2。即计算头和尾的平均MRR

        print("\nCumulative stats are -> ")   # 最后结果的输出
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))

