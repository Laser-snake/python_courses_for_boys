import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# np.random.seed(10)


class Percolation2D():

    def __init__(self, n, p, tests):
        self.N = n
        self.P = p
        self.tests = tests
        self.Nsquare = self.N * self.N
        self.grid = np.zeros((self.N, self.N))

    def random_choise(self):
        for i in range(self.N):
            for j in range(self.N):
                self.grid[i, j] = np.random.choice(
                    (0, 1), 1, p=[1 - self.P, self.P])

    def point_neighborhood(self, i, j):
        indices = []
        for i_ind in range(i-1, i+2):
            for j_ind in range(j-1, j+2):
                if (i_ind == i or j_ind == j):
                    if (i_ind >= 0 and i_ind < self.N):
                        if (j_ind >= 0 and j_ind < self.N):
                            if [i_ind, j_ind] != [i, j] and self.grid[i_ind, j_ind] == 1:
                                indices.append([i_ind, j_ind])
        return indices

    def treasure_map(self):
        graph = {}
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i, j] == 1:
                    graph[(i, j)] = graph.get((i, j), []) + \
                        self.point_neighborhood(i, j)
        return graph

    def bfs(self, start, goal, graph):
        # говно а не поиск не какого соблюдения типов
        start = tuple(start)
        goal = tuple(goal)
        queue = deque([start])
        visited = {start: None}
        currentNode = start

        while queue:
            currentNode = queue.popleft()
            if tuple(currentNode) == goal:
                return True

            nextNodes = graph[tuple(currentNode)]
            for nextNode in nextNodes:
                if tuple(nextNode) not in visited:
                    queue.append(nextNode)
                    visited[tuple(nextNode)] = currentNode
        return False

    def matrix_check(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[0, i] == 1:
                    if self.bfs([0, i], [self.N - 1, j], self.treasure_map()):
                        return True
        return False

    def chalenge(self):
        poz = 0.
        for i in range(self.tests):
            self.random_choise()
            if self.matrix_check():
                poz = poz + 1.
        return poz/float(self.tests)


n = [5, 10, 15]
steps = 0.1
N = int(1.0/steps)
y = np.zeros((len(n), N))

k = 1000
for i in range(len(n)):
    p = -steps
    for j in range(N):
        p = p + steps
        perc = Percolation2D(n[i], p, k)
        y[i][j] = perc.chalenge()
x = np.linspace(0, 1, N)
for i in range(len(n)):
    plt.plot(x, y[i])
plt.show()

perc = Percolation2D(6, 0.5, 100)
perc.random_choise()
a = perc.treasure_map()
c = [0, 0]
b = [0, 5]
print(perc.grid)
# print(perc.bfs(c, b, a))
print(perc.chalenge())
