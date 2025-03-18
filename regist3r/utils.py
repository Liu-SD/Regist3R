import heapq
import torch
from scipy.spatial.transform import Rotation
import numpy as np
import math
from collections import defaultdict
from functools import partial
from croco.utils.misc import MetricLogger as CrocoMetricLogger, SmoothedValue

# from graphviz import Digraph
# from PIL import Image
# import io

############################ Tree functions ############################
class Tree:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.childs = []

    def add_child(self, value):
        """ 添加子节点并返回新节点 """
        new_child = Tree(value, parent=self)
        self.childs.append(new_child)
        return new_child

    def compress_depth(self):
        """ 执行深度压缩：偶数深度的节点连接到祖父节点 """
        # 找到根节点并收集所有需要处理的节点
        root = self.find_root()
        even_depth_nodes = self._collect_even_depth_nodes(root)
        
        for node in even_depth_nodes:
            grandfather = node.parent.parent  # 祖父节点一定存在（因为收集条件保证）
            # 从原父节点中移除
            node.parent.childs.remove(node)
            # 添加到祖父节点
            grandfather.childs.append(node)
            # 更新父指针
            node.parent = grandfather

    def find_root(self):
        """ 找到树的根节点 """
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    def _collect_even_depth_nodes(self, root):
        """ 收集所有深度为偶数且≥2的节点 """
        even_nodes = []
        from collections import deque
        queue = deque([(root, 0)])  # (节点, 深度)
        
        while queue:
            node, depth = queue.popleft()
            # 收集深度≥2的偶数层节点
            if depth >= 2 and depth % 2 == 0:
                even_nodes.append(node)
            # 继续遍历子节点
            for child in node.childs:
                queue.append((child, depth + 1))
        return even_nodes

    def __str__(self):
        """ 美观的树结构打印 """
        return self._print_tree()

    def _print_tree(self, prefix="", is_last=True):
        """ 生成树的可视化字符串 """
        connector = "└── " if is_last else "├── "
        result = prefix + connector + str(self.value) + "\n"
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.childs):
            is_last_child = i == len(self.childs) - 1
            result += child._print_tree(new_prefix, is_last_child)
        
        return result

    # def _add_nodes(self, dot, node):
    #     dot.node(str(id(node)), str(node.value))
    #     for child in node.childs:
    #         self._add_nodes(dot, child)

    # def _add_edges(self, dot, node):
    #     for child in node.childs:
    #         dot.edge(str(id(node)), str(id(child)))
    #         self._add_edges(dot, child)

    # def render_tree(self, format='png'):
    #     dot = Digraph()
    #     self._add_nodes(dot, self)
    #     self._add_edges(dot, self)
    #     image_bytes = dot.pipe(format=format)
        
    #     # 将字节流转换为 PIL 图像
    #     image = Image.open(io.BytesIO(image_bytes))
        
    #     # 将 PIL 图像转换为 numpy.ndarray
    #     image_array = np.array(image)
    #     return image_array

def similarity_to_distance(similarity_matrix):
    return 1 - similarity_matrix

def dijkstra(dist_matrix, start):
    n = len(dist_matrix)
    dist = [float('inf')] * n
    dist[start] = 0
    prev = [None] * n
    pq = [(0, start)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v in range(n):
            if u == v:
                continue
            new_dist = dist[u] + dist_matrix[u][v]
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))
    return prev

def kruskal(dist_matrix, start=0):
    from scipy.sparse.csgraph import minimum_spanning_tree
    tree = minimum_spanning_tree(dist_matrix)
    n = len(dist_matrix)
    parent = [None] * n
    edge_nodes = [start]
    ei, ej = tree.nonzero()
    ei, ej = np.concatenate((ei, ej)), np.concatenate((ej, ei))
    while len(edge_nodes) > 0:
        next_edge_nodes = []
        for edge_node in edge_nodes:
            child = ej[ei == edge_node]
            for c in child:
                if c != start and parent[c] is None:
                    next_edge_nodes.append(c)
                    parent[c] = edge_node
        edge_nodes = next_edge_nodes
    return parent

def _build_tree(parent:list[int]):
    root = [i for i, v in enumerate(parent) if v is None][0]
    root_node = Tree(value=root)
    edge_set = {root : root_node}
    while len(edge_set) > 0:
        next_edge_set = {}
        for i, v in enumerate(parent):
            if v in edge_set:
                i_node = edge_set[v].add_child(i)
                next_edge_set[i] = i_node
        edge_set = next_edge_set
    return root_node

def build_tree(affinity_matrix, tree_type, start=0, compression_factor=0):
    assert tree_type in ['SPT', 'MST']
    distance_matrix = similarity_to_distance(affinity_matrix).cpu().numpy()
    if tree_type == 'SPT':
        parent = dijkstra(distance_matrix, start)
    elif tree_type == 'MST':
        parent = kruskal(distance_matrix, start)
        # parent = prim(distance_matrix, start)
    else:
        raise ValueError(f"Tree type {tree_type} must in [SPT, MST].")
    tree = _build_tree(parent)
    for _ in range(compression_factor):
        tree.compress_depth()
    return tree

def generate_random_rotation(size, device):
    random_rotation = torch.from_numpy(Rotation.random(size).as_matrix()).to(device)
    random_rotation_4x4 = torch.eye(4, device=random_rotation.device).unsqueeze(0).repeat(random_rotation.shape[0], 1, 1)
    random_rotation_4x4[:, :3, :3] = random_rotation
    return random_rotation_4x4

@torch.no_grad()
def procrustes_alignment(A : torch.Tensor, B : torch.Tensor, c : torch.Tensor):

    """
    Solves the weighted Procrustes problem: min ∑c_i ||s*R*A_i + t - B_i||^2
    where R is a rotation matrix, s is a scaling factor, t is a translation vector
    
    Parameters:
        A (np.ndarray): Source point cloud, shape (N, D)
        B (np.ndarray): Target point cloud, shape (N, D) 
        c (np.ndarray): Weights for each point pair, shape (N,)
    
    Returns:
        s (float): Optimal scaling factor
        R (np.ndarray): Optimal rotation matrix, shape (D, D)
        t (np.ndarray): Optimal translation vector, shape (D,)
    """
    # Validate inputs
    assert A.shape == B.shape, "Point clouds must have same dimensions"
    assert len(c) == A.shape[0], "Weights must match point count"
    assert torch.all(c >= 0), "Weights must be non-negative"
    assert A.dim() == B.dim() == 2
    
    total_weight = torch.sum(c)
    if total_weight == 0:
        raise ValueError("All weights cannot be zero")
    
    # 1. Compute weighted centroids
    centroid_A = (A.T @ c) / total_weight
    centroid_B = (B.T @ c) / total_weight
    
    # 2. Center the point clouds
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # 3. Compute weighted covariance matrix
    H = (B_centered * c[:, None]).T @ A_centered
    
    # 4. SVD decomposition
    U, S, Vt = torch.linalg.svd(H)
    
    # 5. Compute rotation matrix
    R = U @ Vt
    
    # Handle reflection case
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    # 6. Compute scaling factor
    a_norm = torch.sum(c * torch.sum(A_centered**2, axis=1))
    s = torch.sum(S) / a_norm if a_norm != 0 else 0.0
    
    # 7. Compute translation
    t = centroid_B - s * (R @ centroid_A)
    
    return s, R, t

class MetricLogger(CrocoMetricLogger):
    def __init__(self, delimiter="\t", window_size=20):
        self.meters = defaultdict(partial(SmoothedValue, window_size=window_size))
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            v = 0. if math.isnan(v) or math.isinf(v) else v
            self.meters[k].update(v)
