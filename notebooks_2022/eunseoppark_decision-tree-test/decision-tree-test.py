#!/usr/bin/env python
# coding: utf-8

# **Modules**

# In[ ]:


from graphviz import *
from sklearn.datasets import load_iris
from sklearn import tree
 
from collections import Counter
from itertools import chain, combinations
 
import ast
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import graphviz
import seaborn as sns


# Node class, 생성자 함수

# In[ ]:


class Node:
    def __init__(self,nodeId, label, isRoot=False,parentNode=None,
               leftNode=None,rightNode=None,isTerminal=False, 
                 attr={}):
        self.nodeId = nodeId ## 노드 식별 아이디
        self.label = label ## 노드 텍스트
        self.attr = attr ## 노드 스타일 속성
        self.isRoot = isRoot ## 루트 노드 여부
        self.parentNode = parentNode ## 부모 마디(노드)
        self.leftNode = leftNode ## 왼쪽 자식 노드(마디)
        self.rightNode = rightNode ## 오른쪽 자식 노드
        self.isTerminal = isTerminal ## 터미널 노드 여부
        self.level = 0 ## 노드가 속한 층


# graphviz의 Digraph() 를 이용하여 시각화해주는 함수 

# In[ ]:


def visualize_tree(tree):
    def add_node_edge(tree, dot=None):
        if dot is None:
            dot = Digraph()
            #name = tree
            dot.node(name = str(tree.nodeId), label = str(tree.label), **tree.attr)

        ## left
        if tree.leftNode:
            dot.node(name=str(tree.leftNode.nodeId),label=str(tree.leftNode.label),
                     **tree.leftNode.attr) 
            dot.edge(str(tree.nodeId), str(tree.leftNode.nodeId),
                     **{'taillabel':"yes",'labeldistance':'2.5'})
            dot = add_node_edge(tree.leftNode, dot)
            
        if tree.rightNode:
            dot.node(name=str(tree.rightNode.nodeId),label=str(tree.rightNode.label),
                     **tree.rightNode.attr)
            dot.edge(str(tree.nodeId), str(tree.rightNode.nodeId),
                    **{'headlabel':" no",'labeldistance':'2'})
            dot = add_node_edge(tree.rightNode, dot)

        return dot
        
    dot = add_node_edge(tree)
    
    return dot


# RGB 색상을 16진수로 바꿔주는 함수

# In[ ]:


def RGBtoHex(vals, rgbtype=1):
    """Converts RGB values in a variety of formats to Hex values.

     @param  vals     An RGB/RGBA tuple
     @param  rgbtype  Valid valus are:
                          1 - Inputs are in the range 0 to 1
                        256 - Inputs are in the range 0 to 255

     @return A hex string in the form '#RRGGBB' or '#RRGGBBAA'
    """

    if len(vals)!=3 and len(vals)!=4:
        raise Exception("RGB or RGBA inputs to RGBtoHex must have three or four elements!")
    if rgbtype!=1 and rgbtype!=256:
        raise Exception("rgbtype must be 1 or 256!")

    #Convert from 0-1 RGB/RGBA to 0-255 RGB/RGBA
    if rgbtype==1:
        vals = [255*x for x in vals]

    #Ensure values are rounded integers, convert to hex, and concatenate
    return '#' + ''.join(['{:02X}'.format(int(round(x))) for x in vals])


# 입력값이 정수인지 아닌지 출력하는 함수

# In[ ]:


def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


# **결정트리 클래스**

# In[ ]:


class DecisionTree:
    def __init__(self, tree_type='classification',):
        tree_types = ['classification','regression']
        assert tree_type in tree_types, f'tree_type must be the one of the {tree_types}'
        self.tree_type = tree_type ## 트리 유형
        self.impurity_measure = None ## 불순도 측도
        self.root = None ## 트리 노드
        self.node_id = 0 ## 노드 아이디
        self.col_names = None ## 칼럼 이름
        self.col_types = None ## 변수 타입
        self.X = None ## train data X
        self.y = None ## train data y
        self.leaf_attr = None ## 끝마디 스타일 속성
    
    def traverseInOrder(self, node):
        res = []
        if node.leftNode != None:
            res = res + self.traverseInOrder(node.leftNode)
        res.append(node)
        if node.rightNode != None:
            res = res + self.traverseInOrder(node.rightNode)
        return res
    
    def getDepth(self, root):
        res = self.traverseInOrder(root)
        res = [abs(node.level) for node in res]
        return max(res)
    
    def getLevel(self, node, counter = 1):
        if node.parentNode is None:
            return counter
        else:
            counter += 1
            counter = self.getLevel(node.parentNode,counter)
        return counter
    
    #변수 타입설정, 유니크한 원소가 15개 이하라면 범주형으로 아니면 연속형으로
    def determineTypeOfCol(self,X, num_unique_values_threshold=15):
        col_types = []
        for col in X.columns:
            unique_values = X[col].unique()
            example_value = unique_values[0]
            
            if (isinstance(example_value, str)) or (len(unique_values) <= num_unique_values_threshold):
                col_types.append('categorical')
            else:
                col_types.append('continuous')
        self.col_types = col_types
    #마디에 포함된 데이터 라벨이 하나인지
    def isPure(self,y):
        if len(np.unique(y)) > 1:
            return False
        return True
    #자식 마디의 불순도 계산
    def impurity(self, left_y, right_y):
        y = self.y
        n = len(left_y)+len(right_y)
        
        if self.impurity_measure == 'chi2':
            try:
                label = np.unique(y)
                contingency_table = dict()
                expected_table = dict()
                for l in label:
                    temp1 = []
                    temp1.append(np.sum(left_y==l))
                    temp1.append(np.sum(right_y==l))
                    contingency_table[l] = temp1
                    temp2 = []
                    temp2.append((np.sum(left_y==l) + np.sum(right_y==l))*len(left_y)/n)
                    temp2.append((np.sum(left_y==l) + np.sum(right_y==l))*len(right_y)/n)
                    expected_table[l] = temp2

                observed = np.array([v for k,v in contingency_table.items()]).flatten()
                expected = np.array([v for k,v in expected_table.items()]).flatten()
                impurity_val = np.nansum(np.square(observed-expected)/expected)
            except RuntimeWarning:
                raise
        else:
            pl, pr = len(left_y)/n, len(right_y)/n
            impurity_val = pl*self.individualImpurity(left_y)+                            pr*self.individualImpurity(right_y)
        return impurity_val
    #개별 마디에 포함된 불순도를 계산        
    def individualImpurity(self, y):
        if self.impurity_measure == 'entropy':
            return self._entropy(y)
        elif self.impurity_measure == 'gini':
            return self._gini(y)
        elif self.impurity_measure == 'mse':
            return self._mse(y)
    #엔트로피 계산
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    #지니 지수 계산
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return np.sum([p*(1-p) for p in ps if p > 0])
   #제곱 평균 오차 계산 
    def _mse(self, y):
        if len(y) == 0:
            return 0
        mse = np.mean(np.square(y-np.mean(y)))
        return mse
    #끝 노드 생성
    def createLeaf(self, y, tree_type):
        if tree_type == 'classification':
            classes, counts = np.unique(y, return_counts=True)
            index = counts.argmax()
            return classes[index]
        else:
            return np.mean(y)
    #변수가 범주형일 때 공집합과 전체 집합을 제외한 모든 부분 집합을 구해준다.
    def powerset_generator(self, i):
        for subset in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):
            yield set(subset)
    #변수가 범주형일 때 분리 기준 후보를 만들어준다.    
    def splitSet(self, x):
        ps = [i for i in self.powerset_generator(x) if i != set() and len(i) != len(x)]
        idx = int(len(ps)/2)
        split_set = []
        for j in range(idx):
            split_set.append(tuple(ps[j]))
        return split_set
    #모든 변수에 대한 분리 기준 후보를 생성    
    def getPotentialSplits(self,X):
        potential_splits = {}
        col_types = self.col_types
        for col_idx in range(X.shape[1]):
            unique_value = np.unique(X[:,col_idx])
            if col_types[col_idx] == 'continuous':
                potential_splits[col_idx] = unique_value
            else:
                potential_splits[col_idx] = self.splitSet(unique_value)
        return potential_splits
    #해당 변수와 분리 기준을 이용하여 왼쪽마디,오른쪽 마디에 각각 들어갈 데이터 인덱스를 리턴한다. 
    def split(self, X, col_idx, threshold):
        X_col = X[:,col_idx]
        col_types = self.col_types
        if col_types[col_idx] == 'continuous':
            left_idx = np.argwhere(X_col<=threshold).flatten()
            right_idx = np.argwhere(X_col>threshold).flatten()
        else:
            left_idx = np.argwhere(np.isin(X_col,threshold)).flatten()
            right_idx = np.argwhere(~np.isin(X_col,threshold)).flatten()
        return left_idx, right_idx
    #getPotential_splits 가 생성한 분리 기준을 통하여 최적 변수와 그에 대응하는 본리기준, 불순도 측도 값을 리턴한다.
    def determinBestSplit(self, X, y, potential_splits):
        best_split_column, best_split_value, opt_impurity = '', '', ''
        if self.impurity_measure in ['entropy','gini','mse']: 
            opt_impurity = np.infty
            for col in potential_splits:
                for val in potential_splits[col]:
                    left_idx, right_idx = self.split(X,col,val)
                    cur_impurity = self.impurity(y[left_idx],y[right_idx])
                    if cur_impurity <= opt_impurity:
                        opt_impurity = cur_impurity
                        best_split_column = col
                        best_split_value = val
        else:
            opt_impurity = -np.infty
            for col in potential_splits:
                for val in potential_splits[col]:
                    left_idx, right_idx = self.split(X,col,val)
                    cur_impurity = self.impurity(y[left_idx],y[right_idx])
                    if cur_impurity >= opt_impurity:
                        opt_impurity = cur_impurity
                        best_split_column = col
                        best_split_value = val

        return best_split_column, best_split_value, opt_impurity
    #의사결정트리 모형을 적합 시키는 함수    
    def fit(self,X,y,impurity_measure='entropy',min_sample=5, max_depth=5, 
            type_of_col=None, auto_determine_type_of_col=True,
            num_unique_values_threshold = 15
           ):
        '''
        impurity_measure : 불순도 측도
        min_sample : 노드가 포함해야하는 최소 샘플 개수,
        max_depth : 나무 최대 깊이 설정
        type_of_col : 변수 타입 리스트
        auto_determine_type_of_col : 변수 타입 자동 생성 여부
        num_unique_values_threshold : 범주형으로 지정할 최대 유니크 원소 개수
        '''
        self.X = X
        self.y = y
        ### 랜덤으로 칼럼 선택하는 것도 고르자. X = X[random_indices,:]
#         if type_of_col is None:
#             type_of_col = determinTypeOfCol(X)
        if auto_determine_type_of_col:
            self.determineTypeOfCol(X, num_unique_values_threshold)
        else:
            if type_of_col is None:
                raise ValueError('When auto_determine_type_of_col is False, then type_of_col must be specified')
            assert X.shape[1] == len(type_of_col), 'type_of_col has the same length of X columns'
            give_type_of_col = list(set(type_of_col))
            for toc in give_type_of_col:
                if toc != 'categorical' and toc != 'continuous':
                    raise ValueError('type_of_col must contain categorical or continuous')
            self.col_types = type_of_col
            
        tree_type = self.tree_type
        impurity_measures = ['entropy','gini','chi2'] if tree_type == 'classification' else ['mse']
        assert impurity_measure in impurity_measures,                f'impurity_measure must be the one of the {impurity_measures}'
        self.impurity_measure = impurity_measure
        tree_type = self.tree_type
        self.root = self._growTree(X,y,tree_type,min_sample=min_sample, max_depth=max_depth)
        
        ### assign node a style
        iod = self.traverseInOrder(self.root)
        root_node = [node for node in iod if node.nodeId == 1][0]
        root_node.isRoot = True
        ## set node level
        for nd in iod:
            nd.level = self.getLevel(nd)
        
        colors = sns.color_palette('hls', self.getDepth(self.root))
        
        ## set node level
        if tree_type == 'classification':
            leaf_color = sns.color_palette('pastel', len(np.unique(y)))
#             class_to_color = dict()
#             for i, l in enumerate(np.unique(y)):
#                 class_to_color[l] = RGBtoHex(leaf_color[i])
            leaf_attr = dict()
            for i, l in enumerate(np.unique(y)):
                leaf_attr[l] = {'shape':'box', 'color':f'{RGBtoHex(leaf_color[i])}', 
                                       'fontcolor':f'{RGBtoHex(leaf_color[i])}','peripheries':'2'}
            self.leaf_attr = leaf_attr
        for l in range(1,self.getDepth(self.root)+1):
            color = RGBtoHex(colors[l-1])
            for nd in iod:
                if nd.level == l:
                    if nd.isTerminal:
                        if tree_type == 'classification':
                            nd.attr =  leaf_attr[nd.label]
#                             nd.attr = {'shape':'box', 'color':f'{class_to_color[nd.label]}', 
#                                        'fontcolor':f'{class_to_color[nd.label]}','peripheries':'2'}
                        else:
                            nd.attr = {'shape':'box','peripheries':'2'}
                    else:
                        nd.attr = {'shape':'box','style':'filled',
                                   'fillcolor':f'{color}'}
        
    def _growTree(self, X, y, tree_type, counter=0, min_sample=5, max_depth=5): ## Tree 배고 노드 클래스만 가지고 해야겠다.
        self.node_id += 1
        
        if counter == 0:
            global col_names
#             col_types = self.col_types
            col_names = X.columns
            self.col_names = X.columns
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
        else:
            X = X
        if (self.isPure(y)) or (len(y) <= min_sample) or (counter == max_depth):
            leaf = self.createLeaf(y, tree_type)
            if isinstance(leaf, float):
                if not leaf.is_integer():
                    leaf = round(leaf,2)
            return Node(self.node_id, label=leaf, isTerminal=True)

        else:
            counter += 1
            potential_splits = self.getPotentialSplits(X)
            best_split_column, best_split_value, opt_impurity =                self.determinBestSplit(X, y, potential_splits)
            opt_impurity = round(opt_impurity,4)
            left_idx, right_idx = self.split(X,best_split_column,best_split_value)
            
            ## check for empty data
            if len(left_idx) == 0 or len(right_idx) == 0:
                leaf = self.createLeaf(y, tree_type)
                if isinstance(leaf, float):
                    if not leaf.is_integer():
                        leaf = round(leaf,2)
                return Node(self.node_id, label=round(leaf,4), isTerminal=True)
            
            total_sample = len(y)
            
            col_name = col_names[best_split_column]
#             if self.tree_type == 'classification':
            if self.col_types[best_split_column] == 'continuous':
                question = f'{col_name} <= {best_split_value}\n'+                            f'{self.impurity_measure} : {opt_impurity}\n'+                            f'Samples : {total_sample}'
            else:
                question = f'{col_name} in {best_split_value}\n'+                            f'{self.impurity_measure} : {opt_impurity}\n'+                            f'Samples : {total_sample}'
#             sub_tree = {question:[]}
            node = Node(self.node_id, label=question)
            
            left_child = self._growTree(X[left_idx,:],y[left_idx],tree_type,counter, min_sample, max_depth)  
            right_child = self._growTree(X[right_idx,:],y[right_idx],tree_type,counter, min_sample, max_depth)
            

            if left_child.label == right_child.label:
                node = left_child
            else:
                node.leftNode = left_child
                node.rightNode = right_child
                left_child.parentNode = node
                right_child.parentNode = node

            return node
    
    def predict(self,X):
        return np.array([self._traverse_tree(x, self.root) for _, x in X.iterrows()])
    
    def _traverse_tree(self, x, node):
        if node.isTerminal:
            if isinstance(node.label, str):
                node.label = node.label.replace('\n','')
            return node.label
        
        question = node.label.split('\n')[0]
        
        if ' <= ' in question:
            col_name, value = question.split(' <= ')
            if x[col_name] <= float(value):
                return self._traverse_tree(x, node.leftNode)
            return self._traverse_tree(x, node.rightNode)
        else:
            col_name, value = question.split(' in ')
            if x[col_name] in ast.literal_eval(value):
                return self._traverse_tree(x, node.leftNode)
            return self._traverse_tree(x, node.rightNode)
        
    def pruning(self, node, X_val, y_val):
        X = self.X
        y = self.y
        if isinstance(y, pd.Series):
            y = y.values
        return self._pruning(node, X, y, X_val, y_val)
    
    def _filterX(self, X, node):
        question = node.label.split('\n')[0]
        if ' <= ' in question:
            col_name, value = question.split(' <= ')
            yes_index = X.loc[X[col_name] <= float(value)].index
            no_index = X.loc[X[col_name] > float(value)].index
        else:
            col_name, value = question.split(' in ')
            yes_index = X.loc[X[col_name].isin(ast.literal_eval(value))].index
            no_index = X.loc[~X[col_name].isin(ast.literal_eval(value))].index
        return yes_index, no_index
    
    def _pruning_leaf(self, node, X, y, X_val, y_val):
        if self.tree_type == 'classification':
            classes, counts = np.unique(y, return_counts=True)
            index = counts.argmax()
            leaf = classes[index]
            errors_leaf = np.sum(y_val != leaf)
            errors_decision_node = np.sum(y_val != self.predict(X_val)) ##<---self로 바꿔야해
            if errors_leaf <= errors_decision_node:
                if isinstance(leaf, float):
                    if not leaf.is_integer():
                        leaf = round(leaf,2)
                return Node(node.nodeId, label=leaf, isTerminal=True,
                            attr=self.leaf_attr[leaf])
            else:
                return node
        else:
            return
    
    def _pruning(self, node, X, y, X_val, y_val):
#         assert self.root is not None, 'you must fit first'
        X = X.reset_index(drop=True)
        left_child = node.leftNode
        right_child = node.rightNode
        if node.leftNode.isTerminal == True and node.rightNode.isTerminal == True:
            return self._pruning_leaf(node, X, y, X_val, y_val)
        else:
            if self.tree_type == 'classification':
                tr_yes_idx, tr_no_idx = self._filterX(X, node)
                val_yes_idx, val_no_idx = self._filterX(X_val, node)
                
                if node.leftNode.isTerminal == False:
                    left_child = self._pruning(node.leftNode, X.loc[tr_yes_idx], y[tr_yes_idx],
                                  X_val.loc[val_yes_idx], y_val[val_yes_idx])
#                     node.leftNode = left_child
#                     left_child.parentNode = node
                if node.rightNode.isTerminal == False:
                    right_child = self._pruning(node.rightNode, X.loc[tr_no_idx], y[tr_no_idx],
                                  X_val.loc[val_no_idx], y_val[val_no_idx])
                
#                     node.rightNode = right_child
#                     right_child.parentNode = node
#                 question = node.label.split('\n')[0]
                attr = node.attr
                node = Node(node.nodeId, label=node.label, isTerminal=node.isTerminal)
                node.attr = attr
                node.leftNode = left_child
                left_child.parentNode = node
                node.rightNode = right_child
                right_child.parentNode = node
            else:
                return
        return self._pruning_leaf(node, X, y, X_val, y_val)
        


# In[ ]:


#데이터 셋 가져오기
df = pd.read_csv('../input/personal-key-indicators-of-heart-disease/heart_2020_cleaned.csv')

df


# **Yes  No => 1,0 으로 수치화**

# In[ ]:


columns = ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"]
df[columns] = df[columns].apply(lambda x: x.map({'Yes':1,'No':0}))
df


# In[ ]:


df2 = df.copy() #원핫 인코더 데이터 셋에 사용할거야


# In[ ]:


#원핫 인코더
#pandas get_dummies 함수 사용, 문자를 수치화 시켜주지만 관게성을 없애는 가변수로 만든다.
genHealth = pd.get_dummies(df2['GenHealth'],prefix='GenHealth')
ageCategory = pd.get_dummies(df2['AgeCategory'],prefix='AgeCategory')
race = pd.get_dummies(df2['Race'],prefix='Race')
sex = pd.get_dummies(df2['Sex'],prefix='Sex')
diabetic = pd.get_dummies(df2['Diabetic'],prefix='Diabetic')

#가변수로 만든 속성들 추가
df2 = pd.concat([df2,genHealth,ageCategory,race,sex,diabetic],axis=1)

#변경한 기존 속성은 삭제
df2 = df2.drop(df2[['Race','AgeCategory','GenHealth','Sex','Diabetic']],axis=1)

df2


# In[ ]:


df2.info()


# In[ ]:


#라벨 인코더
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['AgeCategory'] = label_encoder.fit_transform(df['AgeCategory'])
df['Race'] = label_encoder.fit_transform(df['Race'])
df['GenHealth'] = label_encoder.fit_transform(df['GenHealth'])
df


# In[ ]:


dib = {'Yes':1.00, 'No':0.00, 'No, borderline diabetes':0.5, 'Yes (during pregnancy)':0.80}
df['Diabetic'] = df['Diabetic'].apply(lambda x: dib[x])
df['Diabetic'] = df['Diabetic'].astype('float')


# In[ ]:


#라벨 인코딩
X = df.drop('HeartDisease', axis = 1) 
y = df['HeartDisease']


# In[ ]:


#원핫 인코딩
X2 = df2.drop('HeartDisease', axis = 1) 
y2 = df2['HeartDisease']


# 심장병이 없는 데이터가 훨씬 많기 때문에 데이터 균형을 맞춘다. 불균형한 데이터 세트는 이상 데이터를 정확히 찾아내지 못할 수 있다는 문제점이 존재

# In[ ]:


#SMOTE 오버샘플링
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42,k_neighbors =5)
X,y = smote.fit_resample(X,y)
X2,y2 = smote.fit_resample(X2,y2)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42) #라벨인코더
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2,test_size = 0.3, random_state = 42) #원핫인코더


# **정규화**

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train2 = scaler.fit_transform(X_train2)
X_test2 = scaler.transform(X_test2)


# **라벨 인코딩으로 전처리한 결과**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
sclf = DecisionTreeClassifier(criterion='entropy', max_depth=39, min_samples_split=3)
sclf = sclf.fit(X_train, y_train)


# In[ ]:


#np.sum(y_test == sclf.predict(X_test))/len(y_test) #0.9135909275685592
sclf.score(X_test,y_test)


# In[ ]:


num = 1
arrDepth = []
while num <70:
    sclf = DecisionTreeClassifier(criterion='entropy', max_depth=num, min_samples_split=3)
    sclf = sclf.fit(X_train, y_train)
    print(num,":",sclf.score(X_test,y_test))
    arrDepth.append(sclf.score(X_test,y_test))
    num = num+1


# In[ ]:


plt.title('max depth , accuracy')
plt.plot(arrDepth)
plt.show


# In[ ]:


num = 20
arrMinSamplesSplit = []
while num > 1:
    sclf = DecisionTreeClassifier(criterion='entropy', max_depth=39, min_samples_split=num)
    sclf = sclf.fit(X_train, y_train)
    print(num,":",sclf.score(X_test,y_test))
    arrMinSamplesSplit.append(sclf.score(X_test,y_test))
    num = num-1


# In[ ]:


plt.title('min samples split , accuracy')
plt.plot(arrMinSamplesSplit)
plt.show


# 지니계수

# In[ ]:


gclf = DecisionTreeClassifier(criterion='gini', max_depth=39, min_samples_split=3)  # 분리 기준 criterion{“gini”, “entropy”}, default=”gini”
gclf  = gclf.fit(X_train, y_train)


# In[ ]:


np.sum(y_test == gclf.predict(X_test))/len(y_test)


# 원핫 인코딩으로 전처리 결과

# In[ ]:


sclf2 = DecisionTreeClassifier(criterion='entropy', max_depth=39, min_samples_split=3)
sclf2 = sclf.fit(X_train2, y_train2)


# In[ ]:


np.sum(y_test2 == sclf.predict(X_test2))/len(y_test2) #0.9139244728421184


# In[ ]:


gclf2 = DecisionTreeClassifier(criterion='gini', max_depth=39, min_samples_split=3)  # 분리 기준 criterion{“gini”, “entropy”}, default=”gini”
gclf2  = gclf2.fit(X_train2, y_train2)


# In[ ]:


np.sum(y_test2 == gclf2.predict(X_test2))/len(y_test2)


# **로지스틱 회귀**

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[ ]:


model.fit(X_train, y_train)
model.score(X_test,y_test)


# **교차 검증**
# 1. 특정 데이터 셋에 대한 과적합 방지
# 2. 더욱 일반화된 모델 생성 가능
# 3. 데이터셋 규모가 적을 시 과소적합 방지
# 
# 단점 : 모델 훈련 및 평가 소요시간 증가 (반복 학습 횟수 증가)
# 
# 라벨인코딩으로 전처리한 데이터셋으로 테스트

# In[ ]:


# 훈련 세트와 테스트 세트
X_train3, X_test3, y_train3, y_test3 = train_test_split(X,y,test_size = 0.2, random_state = 42)
#훈련 세트에서 20%를 검증세트로 다시 분리
X_train3, X_val, y_train3, y_val = train_test_split(X_train3,y_train3,test_size = 0.2, random_state = 42)


# In[ ]:


X_train3.shape


# In[ ]:


X_test3.shape


# In[ ]:


X_val.shape


# 검증 세트를 만드느라 훈련 세트가 줄었는데 훈련세트나 검증 세트가 적으면 불안정할 수 있는데 이럴 때 교차검증을 이용해서 안정적인 검증 점수를 얻고 
# 훈련에 더 많은 데이터를 사용 할 수 있다.

# In[ ]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold #분류 모델에 쓰는 분할기 (훈련 데이터를 섞기 위해서 사용)

dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=39, min_samples_split=3)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X,y,test_size = 0.2, random_state = 42)

scores = cross_validate(dt,X_train4,y_train4,cv=StratifiedKFold()) #기본적으로 5폴드 교차 검증 수행
scores
#fit_time : 훈련하는 시간
#score_time : 검증하는 시간
#test_score : 검증 폴드 점수


# In[ ]:


np.mean(scores['test_score'])

