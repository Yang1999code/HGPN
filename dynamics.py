import numpy as np


class FitzHughNagumo:
    def __init__(self, args, A):
        self.L = A - np.diag(np.sum(A, axis=1)) # Difussion matrix: x_j - x_i
        
        param = args[args.dynamics]
        self.a = param.a
        self.b = param.b
        self.c = param.c
        self.epsilon = param.epsilon
        self.k_in = np.sum(A, axis=1) # in-degree
    
    def f(self, x, t):
        
        node_num = x.shape[0] // 2
        x1, x2 = x[:node_num], x[node_num:]
        
        # x1
        f_x1 = x1 - (x1 ** 3)/3 - x2
        outer_x1 = self.epsilon * np.dot(self.L, 1/self.k_in) # epsilon * sum_j Aij * ((x1_j - x1_i) / k_in_i)
        dx1dt = f_x1 + outer_x1
        
        # x2
        f_x2 = self.a + self.b * x1 + self.c * x2
        outer_x2 = 0.0
        dx2dt = f_x2 + outer_x2
        
        if np.isnan(dx1dt).any():
            print('nan during simulation!')
            exit()
        
        dxdt = np.concatenate([dx1dt, dx2dt], axis=0)
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    

class HindmarshRose:
    def __init__(self, args, A):
        self.A = A   # Adjacency matrix
        
        param = args[args.dynamics]
        self.a = param.a
        self.b = param.b
        self.c = param.c
        self.u = param.u
        self.s = param.s
        self.r = param.r
        self.epsilon = param.epsilon
        self.v = param.v
        self.lam = param.lam
        self.I = param.I
        self.omega = param.omega
        self.x0 = param.x0
    
    def f(self, x, t):
        
        node_num = x.shape[0] // 3
        x1, x2, x3 = x[:node_num], x[node_num:2*node_num], x[2*node_num:]
        mu_xj = 1 / (1 + np.exp(-self.lam * (x1 - self.omega)))
        
        # x1
        f_x1 = x2 - self.a * x1 ** 3 + self.b * x1 ** 2 - x3 + self.I
        outer_x1 = self.epsilon * (self.v - x1) * np.dot(self.A, mu_xj)
        dx1dt = f_x1 + outer_x1
        
        # x2
        f_x2 = self.c - self.u * x1 ** 2 - x2
        outer_x2 = 0.0
        dx2dt = f_x2 + outer_x2
        
        # x3
        f_x3 = self.r * (self.s * (x1 - self.x0) - x3)
        outer_x3 = 0.0
        dx3dt = f_x3 + outer_x3
        
        if np.isnan(dx1dt).any():
            print('nan during simulation!')
            exit()
        
        dxdt = np.concatenate([dx1dt, dx2dt, dx3dt], axis=0)
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    
    
class CoupledRossler:
    def __init__(self, args, A):
        self.L = A - np.diag(np.sum(A, axis=1)) # Difussion matrix: x_j - x_i
        
        param = args[args.dynamics]
        self.a = param.a
        self.b = param.b
        self.c = param.c
        self.epsilon = param.epsilon
        self.delta = param.delta
    
    def f(self, x, t):
        
        node_num = x.shape[0] // 3
        x1, x2, x3 = x[:node_num], x[node_num:2*node_num], x[2*node_num:]
        omega = np.random.normal(1, self.delta, size=node_num)
        
        # x1
        f_x1 = - omega * x2 - x3
        outer_x1 = self.epsilon * np.dot(self.L, x1)
        dx1dt = f_x1 + outer_x1
        
        # x2
        f_x2 = omega * x1 + self.a * x2
        outer_x2 = 0.0
        dx2dt = f_x2 + outer_x2
        
        # x3
        f_x3 = self.b + x3 * (x1 + self.c)
        outer_x3 = 0.0
        dx3dt = f_x3 + outer_x3
        
        if np.isnan(dx1dt).any():
            print('nan during simulation!')
            exit()
        
        dxdt = np.concatenate([dx1dt, dx2dt, dx3dt], axis=0)
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])


#
# import numpy as np
#
# class Kuramoto:
#     def __init__(self, args, A):
#         """
#         Kuramoto Model Initialization.
#         Args:
#             args: 参数配置，包含内禀频率 (omega) 和耦合强度 (epsilon)。
#             A: 邻接矩阵，表示网络结构。
#         """
#         self.A = np.array(A)  # 邻接矩阵
#         self.L = self.A - np.diag(np.sum(self.A, axis=1))  # 拉普拉斯矩阵
#
#         param = args["Kuramoto"]
#         self.omega = np.array(param["omega"])  # 节点的内禀频率
#         self.epsilon = param["epsilon"]  # 耦合强度
#
#     def f(self, x, t):
#         """
#         Compute dx/dt for Kuramoto model.
#         Args:
#             x: 当前相位 (数组)，形状为 (节点数,)。
#             t: 当前时间 (标量)，可选参数用于与其他模型接口一致。
#         Returns:
#             dxdt: 每个节点的相位变化率，形状为 (节点数,)。
#         """
#         node_num = x.shape[0]
#
#         # Intrinsic dynamics: 内禀频率对每个节点的贡献
#         f_x = self.omega
#
#         # Coupling dynamics: 节点间通过邻接关系的耦合影响
#         coupling = self.epsilon * np.sum(
#             self.A * np.sin(np.outer(x, np.ones(node_num)) - np.outer(np.ones(node_num), x)),
#             axis=1
#         )
#
#         dxdt = f_x + coupling
#         return dxdt
#
#     def g(self, x, t):
#         """Inherent noise (diagonal noise matrix)."""
#         return np.diag([0.0] * x.shape[0])
