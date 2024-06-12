from gensim.models import Word2Vec as GensimWord2Vec
import networkx as nx
import random
import numpy as np

class MetaPath2Vec:
    def __init__(self, G, metapath, dimensions=64, walk_length=30, num_walks=200, workers=4):
        self.G = G
        self.metapath = metapath
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.model = None

    def generate_walks(self):
        walks = []
        for _ in range(self.num_walks):
            for node in self.G.nodes():
                if self.G.nodes[node].get('type') == self.metapath[0]:  # 确保起始节点类型匹配元路径的第一个类型
                    walk = self.random_walk(self.walk_length, start_node=node)
                    if walk:
                        walks.append(walk)
        print(f"Total generated walks: {len(walks)}")
        return walks

    def random_walk(self, length, start_node):
        walk = [start_node]
        current_node = start_node
        for _ in range(length - 1):
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                break
            next_node_type = self.metapath[(self.metapath.index(self.G.nodes[current_node]['type']) + 1) % len(self.metapath)]
            next_node = random.choice([n for n in neighbors if self.G.nodes[n]['type'] == next_node_type])
            if not next_node:
                break
            walk.append(next_node)
            current_node = next_node
        return walk

    def fit(self, window=10, min_count=1, sg=1, iter=5):
        walks = self.generate_walks()
        print("Generated walks sample:", walks[:2])  # 打印前两个游走路径以供检查
        self.model = GensimWord2Vec(walks, vector_size=self.dimensions, window=window, min_count=min_count, sg=sg,
                                    workers=self.workers, epochs=iter)

        # 为未出现在游走中的节点初始化嵌入向量
        for node in self.G.nodes():
            if node not in self.model.wv:
                # 使用numpy创建一个零向量
                zero_vector = np.zeros(self.model.vector_size, dtype=np.float32)
                self.model.wv[node] = zero_vector  # 正确的方式是使用numpy数组

    def get_node_embedding(self, node_id):
        if node_id in self.model.wv:
            return self.model.wv[node_id]
        else:
            raise ValueError(f"The node '{node_id}' is not in the vocabulary. Please check if it was included in the random walks.")