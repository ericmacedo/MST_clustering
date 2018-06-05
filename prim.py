import numpy as np
import time
from heapq import *
from sklearn.metrics import silhouette_score
from random import shuffle


class Prim:
    def __init__(self, dataset: np.array, k: int):
        start = time.time()

        # DATASET
        self.dataset = dataset
        self.n = dataset.shape[0]
        self.k = k

        self.classes = np.repeat(-1, self.n)  # CLASSES VECTOR

        ''' BINARY HEAP (heapq)
            PYTHON'S IMPLEMENTATION OF A BINARY HEAP
            https://docs.python.org/2/library/heapq.html
            IT OPERATES ON SIMPLE LIST STRUCTURES
        '''
        self.weightedQueue = []  # EMPTY QUEUE
        for i in range(self.n):  # O(n)
            '''
                PUSHES THE INFINITE WEIGHT TO EACH VERTICE ON THE GRAPH
                [0] -> DISTANCE FROM [1] TO [2]
                O(n log n)
            '''
            heappush(self.weightedQueue, [np.inf, i, i])  # O(log n)

        '''
            SHUFLES THE QUEUE, ORTHERWISE IT WILL ALWAYS START AT THE Q[0]
            O(n log n)
        '''
        shuffle(self.weightedQueue)

        self.visited = [False for i in range(0, self.n)]  # O(n)
        self.S = []  # THE MST VECTOR

        self.start()

        # DATA ANALYSIS
        self.silhouette = silhouette_score(dataset, self.classes, metric='euclidean')

        end = time.time()
        self.time = end - start

    # CALCULATES THE EUCLIDEAN DISTANCE OF X[] AND Y[]
    def euclidean(self, x: np.array, y: np.array):
        return np.sqrt((np.sum((x - y)**2)))

    def calculate_distance(self, item: list):
        '''
            @param item: RECEIVE A QUEUE ITEM AS PARAMETER
            MARK THE [2] AS VISITED
            CALCULATE THE DISTANCES FROM [2] TO ALL THE NON VISITED VERTICES
            AND PUSHES IT TO THE QUEUE
            O(n)
        '''
        self.visited[item[2]] = True
        for i in range(0, self.n):
            if(self.visited[i]):  # IF VISITED -> IGNORE IT
                continue
            distance = self.euclidean(self.dataset[item[2]], self.dataset[i])
            heappush(self.weightedQueue, [distance, item[2], i])

    def start(self):
        '''
            THIS FIRST VERTICE IS A SPECIAL CASE, IT IS MANIPULATED
            DIFFERENTLY FROM THE NEXT VERTICES
        '''
        iterator = heappop(self.weightedQueue)
        self.calculate_distance(iterator)

        iterator = heappop(self.weightedQueue)

        '''
            O(n²)
        '''
        while(self.weightedQueue):  # WHILE QUEUE IS NOT EMPTY
            '''
                IF [2] NOT VISITED
                APPEND IT TO THE MST VECTOR
                CALCULATE NOT VISITED NEIGHBORS DISTANCE
                POP ITEM FROM THE QUEUE
                O(n)
            '''
            if(not self.visited[iterator[2]]):
                self.S.append(iterator)
                self.calculate_distance(iterator)
            iterator = heappop(self.weightedQueue)

        '''
            kmax STORES THE INDEXES OF THE LARGESTS EDGES
            FIRST THE MST VECTOR IS PUSHED INTO kmax WITH THE heappush
            METHOD
            THEM, kmax IS HEAPSORTED AND ITS K - 1 LAST ELEMENTS ARE STORED
            O(n log n)
        '''
        kmax = []
        for i in range(len(self.S)):  # O(n)
            heappush(kmax, [self.S[i][0], i])  # O(n log n)

        # O(n log n)
        kmax = [heappop(kmax)[1] for i in range(len(kmax))][-self.k + 1:]

        classe = 0
        for i in range(len(self.S)):  # O(n)
            '''
                CLASSIFICATE THE MST BY NAVIGATING
                IT IS AN ADAPTATION OF THE WIDTH SEARCH ON A GRAPH
            '''
            if(i in kmax):
                classe += 1
                self.classes[self.S[i][2]] = classe
            elif(self.classes[self.S[i][1]] == -1):
                classe += 1
                self.classes[self.S[i][1]] = classe
                self.classes[self.S[i][2]] = classe
            else:
                self.classes[self.S[i][2]] = self.classes[self.S[i][1]]
