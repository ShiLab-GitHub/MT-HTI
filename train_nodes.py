from model import MetaPath2Vec
import pickle
import networkx as nx

def main():
    # 加载图结构
    G = nx.read_gml('G.gml')
    # 定义元路径，这里假设有两个类型的节点 'herb' 和 'target'
    metapath = ['herb', 'target', 'pathway', 'target', 'herb', 'efficacy']

    # 初始化并训练模型
    mp2v = MetaPath2Vec(G, metapath, dimensions=128, walk_length=10, num_walks=7, workers=4)
    mp2v.fit(window=30, min_count=5, sg=1, iter=11)

    # 保存模型到文件
    with open("h-t-p-t-h-e.pkl", "wb") as f:
        pickle.dump(mp2v, f)
    print("Metapath2Vec model trained and saved successfully.")

if __name__ == "__main__":
    main()