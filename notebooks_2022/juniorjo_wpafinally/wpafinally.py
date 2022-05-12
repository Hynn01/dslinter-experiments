#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.listdir('../input/swarm-intelligence')


# In[ ]:


# import sys
# sys.path.append('../swarm-intelligence')
# sys.path.append('../swarm-intelligence/Utils')
# sys.path.append("../input/swarm-intelligence/Utils/Assist.py")
# sys.path


# In[ ]:


cd ../input/swarm-intelligence/Utils


# In[ ]:





# In[ ]:


import numpy as np
# import sys
# sys.path.append("./Utils")
np.set_printoptions(threshold=np.inf)

import Assist
import Standard
import community as community_louvain
import copy
import pandas as pd
import Algorithm
import igraph as ig

POP_SIZE = 100  # 每一代狼群数量
# 头狼仅有一个
# 探狼占比0.19
# 猛狼占比0.8
Tan_Rate = 0.19
Meng_Rate = 0.8
TMAX = 3  # 探狼最大游走次数

# 小网络地时候，比率大一些，karate&dolphins
Step_Warder_Rate = 0.4  # 游走时候替代边的比率
Step_Call_Rate = 0.8  # 召唤时候替代边的比率
Step_Besiege_Rate = 0.2  # 围攻时替代边的比率

# 适应度函数为exp-（Q-1）
# 猛狼被召唤距离头狼适应度小于NEAR时候，发起围攻行为
NEAR = 0.1

#大网络地时候，比率小一些，football&polbooks
# Step_Warder_Rate = 0.2      #游走时候替代边的比率
# Step_Call_Rate = 0.4        #召唤时候替代边的比率
# Step_Besiege_Rate = 0.1     #围攻时替代边的比率

# 以上替代边仅为初始节点不变，被连节点变，每次替代的时候,有百分之10的概率，初始节点也变化
MUTATION_RATE = 0.1  # 变异比率

ELISTISM_RATE = 0.1  # 第上一轮次的

N_GENERATIONS = 50  # 迭代轮次


def ECList(initadjacencyMatrix):
    # print(init_pop)
    dimension = initadjacencyMatrix.shape[0]  # 维度为节点数量
    sample = np.arange(dimension)  # 选取的样本池
    # nn = initG.edges
    totalExistLedge = [[i] for i in range(dimension)]  # m*n  每行第一个为该结点标号(id) 其余为邻居结点id
    totalToConnectLedge = [[] for _ in range(dimension)]  # m*(62-n) 未连接的边
    totalExistLedgeNumber = []  # 每个点已经连接到的边的数量
    for i in range(dimension):
        for j in range(i + 1, dimension):
            if initadjacencyMatrix[i][j] == 1:
                # 对应的id为i,j的结点邻居均增加
                totalExistLedge[i].append(j)
                totalExistLedge[j].append(i)

    for i in range(dimension):
        # 获取未获取边（可以重连的目标边）
        totalToConnectLedge[i] = list(set(sample) - set(totalExistLedge[i]))
        # 获取未获取边（可以重连的目标边）
        totalExistLedgeNumber.append(len(totalExistLedge[i]) - 1)

    return totalExistLedge, totalToConnectLedge, totalExistLedgeNumber


def InitPop(initadjacencyMatrix, T, POP_SIZE=100):
    # 初始化种群 0 1 一行是一个个体，行数为种群大小  x与y共用一行 6*2*2（DNA_SIZE*2）
    init_pop = np.random.randint(1, size=(POP_SIZE, T * 4))  # matrix,相当于np.zeros((POP_SIZE, DNA_SIZEp))
    dimension = initadjacencyMatrix.shape[0]  # 维度为节点数量
    totalExistLedge, totalToConnectLedge, totalExistLedgeNumber = ECList(initadjacencyMatrix)

    indi = 0
    while indi < POP_SIZE:
        # 每个基因
        for gene in range(T):

            # 被选择的节点补充到基因中
            fnode = np.random.randint(dimension)
            # 保证结点有连边,且并未与其他节点都有连边
            while not 0 < totalExistLedgeNumber[fnode] < dimension:
                fnode = np.random.randint(dimension)  # 被选择的节点补充到基因中

            # 0，2位均为起始点，起始点为自己
            init_pop[indi][0 + gene * 4] = totalExistLedge[fnode][0]
            init_pop[indi][2 + gene * 4] = totalExistLedge[fnode][0]

            # 该结点将删除的边,totalExistLedge[fnode][0]是该节点本身
            deleteLedge = np.random.randint(1, len(totalExistLedge[fnode]))
            init_pop[indi][1 + gene * 4] = totalExistLedge[fnode][deleteLedge]
            # 该结点将连接的边
            addLedge = np.random.randint(0, len(totalToConnectLedge[fnode]))
            init_pop[indi][3 + gene * 4] = totalToConnectLedge[fnode][addLedge]

        # 该基因满足条件再去构建下一个基因
        if checkOk(init_pop[indi]): indi += 1

    return init_pop


# 查看基因是否满足要求
# 1.不许有重复的连边，重复的拆边
# 2.拆掉此边不可以同时有连接此边

# 3.修改后是否存在孤立节点，先不考虑，看看最后的情况再说
def checkOk(target):
    # print(target)
    # 基因数量
    geneNumbers = len(target) // 4
    halfGeneList = []
    for i in range(geneNumbers):
        # 删除部分,排序后便于比较验证
        halfGeneList.append(sorted(target[i * 4:i * 4 + 2]))
        # 连接部分,排序后便于比较验证
        halfGeneList.append(sorted(target[i * + 2:i * 4 + 4]))

    edgeNumber = len(halfGeneList)
    for i in range(edgeNumber):
        for j in range(i + 1, edgeNumber):
            # 出现两组一样的连边,无论是重复连边、重复删边、亦或是删了又连边都是可以返回满足要求的
            if halfGeneList[i] == halfGeneList[j]:
                return False
    return True


def fixMatrix(tip, adj, T):
    # 深拷贝，不能修改初始的邻接矩阵
    a = copy.deepcopy(adj)
    for i in range(T):
        # 拆边
        a[tip[0 + i * 4]][tip[1 + i * 4]] = 0
        a[tip[1 + i * 4]][tip[0 + i * 4]] = 0
        # 连边
        a[tip[2 + i * 4]][tip[3 + i * 4]] = 1
        a[tip[3 + i * 4]][tip[2 + i * 4]] = 1
    return a


# 获取适应度,返回ndarray类对象
def getFitness(pop, adjacencyMatrix, T, alg):
    # np的numpy.ndarray类，可以将整个ndarray当做索引
    fitness = np.zeros(POP_SIZE, )
    init_adj = copy.deepcopy(adjacencyMatrix)

    # 获取每一个个体的适应度
    for i in range(POP_SIZE):
        # 获得最新邻接矩阵
        new_adj = fixMatrix(pop[i], init_adj, T)
        # 获取图
        new_G_nx = Assist.AdjacencyMatrxiToGraph(new_adj)
        new_G = ig.Graph.from_networkx(new_G_nx)
        Partition = alg(new_G)
        # if alg == Algorithm.SCA:
        #
        #     Partition = alg(new_G, K)
        #
        # elif alg == Algorithm.FastNewman or alg == Algorithm.LPA:
        #
        #     Partition = alg(new_G).Run()
        #
        # elif alg == community_louvain.best_partition:
        #
        #     Partition = list(alg(new_G).values())
        # else:
        #     Partition = alg(new_G)

        # 获取这个个体的Q值
        new_Q = Standard.Self_Modularity(new_adj, Partition)
        # 将对应个体的Q值存入对应索引位置
        # 适应度函数为exp-（Q-1）
        fitness[i] = np.exp(-(new_Q - 1))

    return fitness


def getFitnessSingle(single, adjacencyMatrix, T, alg):
    init_adj = copy.deepcopy(adjacencyMatrix)
    # 获得最新邻接矩阵
    new_adj = fixMatrix(single, init_adj, T)
    # 获取图
    new_G_nx = Assist.AdjacencyMatrxiToGraph(new_adj)
    new_G = ig.Graph.from_networkx(new_G_nx)
    # if alg == Algorithm.SCA:
    #
    #     Partition = alg(new_G, K)
    #
    # elif alg == Algorithm.FastNewman or alg == Algorithm.LPA:
    #
    #     Partition = alg(new_G).Run()
    #
    # elif alg == community_louvain.best_partition:
    #
    #     Partition = list(alg(new_G).values())
    # else:
    #     Partition = alg(new_G)

    # 获取这个个体的Q值
    Partition = alg(new_G)
    new_Q = Standard.Self_Modularity(new_adj, Partition)
    # 将对应个体的Q值存入对应索引位置
    # 适应度函数为exp-（Q-1）
    return np.exp(-(new_Q - 1))


# pop——list
# 返回numpy.ndarray类对象,按顺序排列，适应度最高的为0 最低位POP_SIZE-1

def select(pop, fitness):
    # 返回适应度的大小排序
    bestIndex = np.array(Get_List_Max_Index(fitness, POP_SIZE))
    return pop[bestIndex]


# 游走行为
def Wander(popInit, initadjacencyMatrix, T, alg,TMAX=3):
    pop = copy.deepcopy(popInit)
    # 1-19号位探狼
    dimension = initadjacencyMatrix.shape[0]
    # 用初始矩阵得到的信息便可以，之后可以使用checkOk进行查验
    totalExistLedge, totalToConnectLedge, totalExistLedgeNumber = ECList(initadjacencyMatrix)
    # 头狼的适应度
    maxFitness = getFitnessSingle(pop[0], initadjacencyMatrix, T, alg)
    for tanId in range(1, int(POP_SIZE * Tan_Rate) + 1):

        time = 1  # 游走次数
        curFitness = getFitnessSingle(pop[tanId], initadjacencyMatrix, T, alg)

        Beifen = pop[tanId].copy()  # 第x只探狼的备份
        # 当前探狼的游走次数大于TMAX或者当前探狼的适应度大于头狼，退出循环
        while time > TMAX or curFitness > maxFitness:
            # 选中的基因
            Genes = np.random.choice(T, size=int(T * Step_Warder_Rate), replace=False)
            while True:
                # 每个被选中的基因
                for gene in Genes:

                    # 被选择的节点补充到基因中
                    fnode = np.random.randint(dimension)
                    # 保证结点有连边,且并未与其他节点都有连边
                    while not 0 < totalExistLedgeNumber[fnode] < dimension:
                        fnode = np.random.randint(dimension)  # 被选择的节点补充到基因中

                    # 0，2位均为起始点，起始点为自己
                    Beifen[0 + gene * 4] = totalExistLedge[fnode][0]
                    Beifen[2 + gene * 4] = totalExistLedge[fnode][0]

                    # 该结点将删除的边,totalExistLedge[fnode][0]是该节点本身
                    deleteLedge = np.random.randint(1, len(totalExistLedge[fnode]))
                    Beifen[1 + gene * 4] = totalExistLedge[fnode][deleteLedge]
                    # 该结点将连接的边
                    addLedge = np.random.randint(0, len(totalToConnectLedge[fnode]))
                    Beifen[3 + gene * 4] = totalToConnectLedge[fnode][addLedge]

                # 该基因满足条件才去比对
                if checkOk(Beifen): break

            time += 1  # 游走数加一
            preFitness = curFitness  # 上一轮的curFitness
            curFitness = getFitnessSingle(Beifen, initadjacencyMatrix, T, alg)

            # 如果当前适应度已经大于最大适应度，换！且退出循环
            if curFitness > maxFitness:
                # Beifen一定会被初始化，更换头狼
                pop[tanId], pop[0] = pop[0], Beifen
                # 头狼适应度更新
                maxFitness = curFitness
                break

            # 如果这一次的fitness更大，pop[tanID]更新，继续使用此备份尝试
            if curFitness > preFitness:
                pop[tanId] = Beifen.copy()
                # Beifen = pop[tanId].copy()
            # 如果这一次的fitness小,取pop【tanId】那个较大的
            else:
                Beifen = pop[tanId].copy()
    # 返回新的头狼、探狼以及猛狼
    return pop


def CallUp(popInit, initadjacencyMatrix, T, alg):
    pop = copy.deepcopy(popInit)
    # 1-19号位探狼
    dimension = initadjacencyMatrix.shape[0]
    # 用初始矩阵得到的信息便可以，之后可以使用checkOk进行查验
    totalExistLedge, totalToConnectLedge, totalExistLedgeNumber = ECList(initadjacencyMatrix)
    # 头狼的适应度
    maxFitness = getFitnessSingle(pop[0], initadjacencyMatrix, T, alg)

    for call_ID in range(int(POP_SIZE * Tan_Rate) + 1, POP_SIZE):

        Beifen = pop[call_ID].copy()  # 第x只探狼的备份

        # 选中的基因,对应召唤的比率
        Genes = np.random.choice(T, size=int(T * Step_Call_Rate), replace=False)
        while True:
            # 每个基因
            for gene in Genes:

                # 被选择的节点补充到基因中
                fnode = np.random.randint(dimension)
                # 保证结点有连边,且并未与其他节点都有连边
                while not 0 < totalExistLedgeNumber[fnode] < dimension:
                    fnode = np.random.randint(dimension)  # 被选择的节点补充到基因中

                # 0，2位均为起始点，起始点为自己
                Beifen[0 + gene * 4] = totalExistLedge[fnode][0]
                Beifen[2 + gene * 4] = totalExistLedge[fnode][0]

                # 该结点将删除的边,totalExistLedge[fnode][0]是该节点本身
                deleteLedge = np.random.randint(1, len(totalExistLedge[fnode]))
                Beifen[1 + gene * 4] = totalExistLedge[fnode][deleteLedge]
                # 该结点将连接的边
                addLedge = np.random.randint(0, len(totalToConnectLedge[fnode]))
                Beifen[3 + gene * 4] = totalToConnectLedge[fnode][addLedge]

            # 该基因满足条件才去比对
            if checkOk(Beifen): break

        curFitness = getFitnessSingle(Beifen, initadjacencyMatrix, T, alg)

        # 大于头狼适应度，换！
        if curFitness > maxFitness:
            # Beifen一定会被初始化，更换头狼
            pop[call_ID], pop[0] = pop[0], Beifen
            # 头狼适应度更新
            maxFitness = curFitness
        # 是否进行围攻行为
        elif maxFitness - curFitness < 0.1:
            # 执行围攻行为
            # 选中的基因,对应围攻的比率
            Genes = np.random.choice(T, size=int(T * Step_Besiege_Rate), replace=False)
            while True:
                # 每个基因
                for gene in Genes:

                    # 被选择的节点补充到基因中
                    fnode = np.random.randint(dimension)
                    # 保证结点有连边,且并未与其他节点都有连边
                    while not 0 < totalExistLedgeNumber[fnode] < dimension:
                        fnode = np.random.randint(dimension)  # 被选择的节点补充到基因中

                    # 0，2位均为起始点，起始点为自己
                    Beifen[0 + gene * 4] = totalExistLedge[fnode][0]
                    Beifen[2 + gene * 4] = totalExistLedge[fnode][0]

                    # 该结点将删除的边,totalExistLedge[fnode][0]是该节点本身
                    deleteLedge = np.random.randint(1, len(totalExistLedge[fnode]))
                    Beifen[1 + gene * 4] = totalExistLedge[fnode][deleteLedge]
                    # 该结点将连接的边
                    addLedge = np.random.randint(0, len(totalToConnectLedge[fnode]))
                    Beifen[3 + gene * 4] = totalToConnectLedge[fnode][addLedge]

                # 该基因满足条件才去比对
                if checkOk(Beifen): break
        # 以上两者均不满足，啥也不做，现在的适应度比召唤之前的适应度还小也不关，大也不考虑
        else:
            pass

    return pop


def Elistism(off, adjacencyMatrix, T, alg, ELISTISM_RATE=0.1):

    #新生成百分之十
    newPop = np.array(InitPop(adjacencyMatrix,T,int(POP_SIZE*ELISTISM_RATE)))

    #初始的前百分之九十
    off_fitness = np.array(getFitness(off, adjacencyMatrix, T, alg))
    off_90 = np.array(Get_List_Max_Index(off_fitness, int(POP_SIZE * (1 - ELISTISM_RATE))))

    #下一轮次的种群
    real = np.array(list(off[off_90]) + list(newPop))

    return real


def Get_List_Max_Index(list_, n):
    """
    function：
        计算列表中最大的N个数对应的索引

    Parameters:
        list_ - 要分析的列表(list)
        n - 截取最大的n个数(int)

    Returns:
        n_index - 最大n个数的索引

    Modify:
        2020-11-23
    """
    N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
    return list(N_large.index)[:n]


# In[ ]:


cd ../datasets


# In[ ]:


if __name__ == "__main__":
    # initG = nx.read_gml('./datasets/dolphins.gml')
#     Achoice = int(input("请输入的社团划分算法：\n"
#                          "  Louvain  ————————1\n"
#                          "    GN     ————————2\n"
#                          "  Infomap  ————————3\n"
#                          " Fastgreedy————————4\n"
#                          "   LPA     ————————5\n"))
#     datachoice = int(input("请选择进行操作的数据集：\n"
#                            "karate————————1\n"
#                            "dolphins————————2\n"
#                            "football————————3\n"
#                            "polbooks————————4\n"
#                            # "adjnoun——————————5\n"
#                            ))
    Achoice = 5
    datachoice = 4
    # Cost = int(input("请选择边代价比例(百分比)：\n"))
    data = ""
    autoencoder = ""
    algorithm = ""
    # k = -1
    if Achoice == 1:
        # 确立社团划分算法
        algorithm = Algorithm.Louvain

    elif Achoice == 2:
        # 确立社团划分算法
        algorithm = Algorithm.GN

    elif Achoice == 3:
        algorithm = Algorithm.Infomap

    elif Achoice == 4:
        algorithm = Algorithm.CNM

    elif Achoice == 5:
        algorithm = Algorithm.LPA

    if datachoice == 1:
        data = "karate"
    elif datachoice == 2:
        data = "dolphins"
    elif datachoice == 3:
        data = "football"
    elif datachoice == 4:
        data = "polbooks"
    elif datachoice == 5:
        data = "adjnoun"
    # 获取图，真实分组，邻接矩阵
    initG_nx, GTV, adjacencyMatrix = Assist.read_gml(f'{data}.gml')

    initG = ig.Graph.from_networkx(initG_nx)
    # print("Q:", Standard.Self_Modularity(adjacencyMatrix,GTV))
    # dolphins Q = 0.3735  football ：Q = 0.5539  #真实的划分却比louvain社团划分后的Q的值要小，
    # 应该是louvain就是实现问题 后续考虑一下

    # dolphinsQ = 0.5188 football Q =  0.6042
    result = []
    temp = list(range(6))
    save_A = copy.deepcopy(adjacencyMatrix)
    Cost = 0
    while Cost < 10:
        # 向上取整
        Cost += 2
        # T为替换边的数量
        T = TrueCost = int(len(initG_nx.edges) * Cost / 100) + 1
        # list对象->ndarray对象
        # 五次一个平均
        for i in range(5):
            print(f"Cost:{Cost}   ",f"第{i+1}次平均")
            # 返回ndarray类对象
            ParentPop = np.array(InitPop(adjacencyMatrix, T))

            # ExceptionFlag = False
            # while not ExceptionFlag:
            #     try:
                # for _ in range(N_GENERATIONS):
            for _ in range(100):
                # 返回ndarray类对象，这是初始化时的适应度
                fitness = getFitness(ParentPop, adjacencyMatrix, T, algorithm)

                # 输入是ndarray类对象，输出也是,获取头狼、探狼、猛狼的排序
                SelectedPop = select(ParentPop, fitness)

                # 输入是ndarray类对象，输出也是，使用1-19下标位置
                WaitForCallPop = Wander(SelectedPop, adjacencyMatrix, T, algorithm)
                # 使用20-99下标位置，猛狼被召唤并根据情况是否进行围攻
                endPop = CallUp(WaitForCallPop, adjacencyMatrix, T, algorithm)

                OffSpringPop = Elistism(endPop, adjacencyMatrix, T, algorithm)
                # 子辈成父辈
                ParentPop = OffSpringPop

            # 获得适应度最大的个体的索引
            last = getFitness(ParentPop, adjacencyMatrix, T, algorithm)
            index = np.argmax(last)

            # 获得最新邻接矩阵
            new_adj = fixMatrix(ParentPop[index], adjacencyMatrix, T)
            # 获取图
            new_G_nx = Assist.AdjacencyMatrxiToGraph(new_adj)

            new_G = ig.Graph.from_networkx(new_G_nx)
            # Partition = list(community_louvain.best_partition(new_G).values())
            # new_Q = Standard.Self_Modularity(new_adj, Partition)

            # if Achoice == 1:
            #     f1 = algorithm(initG)
            #     initPartition = list(f1.values())
            # elif Achoice == 2:
            #     initPartition = algorithm(initG)
            # elif Achoice == 3:
            #     initPartition = algorithm(initG, k)
            # # FN&LPA 4-5
            # else:
            #     initPartition = algorithm(initG).Run()
            initPartition = algorithm(initG)

            Partition = algorithm(new_G)
            temp[0] = Standard.getNMI(GTV, initPartition)
            temp[1] = Standard.getARI(GTV, initPartition)
            temp[2] = Standard.Self_Modularity(adjacencyMatrix, initPartition)

            temp[3] = Standard.getNMI(GTV, Partition)
            temp[4] = Standard.getARI(GTV, Partition)
            temp[5] = Standard.Self_Modularity(new_adj, Partition)
            result.append(temp.copy())

            # print("初始Q:", Standard.Self_Modularity(adjacencyMatrix, initPartition))
            # print("初始NMI:", Standard.getNMI(GTV, initPartition))
            # print("初始ARI：", Standard.getARI(GTV, initPartition))
            #
            # print("结果Q:", new_Q)
            # print("结果NMI：", Standard.getNMI(GTV, Partition))
            # print("结果ARI：", Standard.getARI(GTV, Partition))
            if np.random.randint(10) < 2:
                Assist.drawNet(new_G_nx, Partition)
                # except Exception as e:
                #     print(e)
                #     ExceptionFlag = False
                # else:
                #     ExceptionFlag = True

#     Assist.Writexcel(result, "./Result/WPA.xlsx", "hhh")


# In[ ]:


cd /kaggle/working


# In[ ]:


ans = pd.DataFrame(result)
ans.to_csv('./Answer.csv',index = False)

