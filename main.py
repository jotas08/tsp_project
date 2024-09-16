import re
import numpy as np
import matplotlib.pyplot as plt

#Usar Matplotlib e ReportLab para visualização do projeto
#matriz de distancia


class TSP:
    def __init__(self, **args):
        self.name = None
        self.type = None
        self.dimension = 0
        self.coords = []
        self.tour = []

        if 'filename' in args:
            self.filename = args['filename']
            self.read_file(args['filename'])
        elif 'dimension' in args and 'coords' in args:
            self.dimension = args['dimension']
            self.coords = np.array(args['coords'])
        else:
            raise ValueError('Invalid arguments')
        
        self.coords = np.array(self.coords)
        self.distance_matrix = self.calculate_distance_matrix()
        # self.total_distance = self.calculate_total_distance()

    def read_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Algumas instancias estao formatadas com espaços entre os dois pontos, então
            # vamos "pegá-las" como "chave e valor", por isso o regex
            match = re.match(r'(\w+)\s*:\s*(.*)', line)
            if match:
                key, value = match.groups()
                if key == 'NAME':
                    self.name = value
                elif key == 'TYPE':
                    self.type = value
                elif key == 'DIMENSION':
                    self.dimension = int(value)
            elif line == 'NODE_COORD_SECTION':
                continue
            elif line == 'EOF':
                break
            else:
                parts = line.split()
                if len(parts) == 3:
                    node, x, y = parts
                    self.tour.append(int(node) - 1)
                    self.coords.append((float(x), float(y)))

        self.coords = np.array(self.coords)
        # self.coords = np.sqrt(((self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :])**2).sum(axis=2))

    def plot_tsp(self,method, fobj):
        x = [self.coords[i][0] for i in self.tour] + [self.coords[self.tour[0]][0]]
        y = [self.coords[i][1] for i in self.tour] + [self.coords[self.tour[0]][1]]

        plt.figure(figsize=(18, 18))
        plt.plot(x, y, marker='o', linestyle='-', color='b')

        for i in range(len(self.tour)):
            plt.text(x[i], y[i], f'{self.tour[i] + 1}', fontsize=12, ha='right')
        
        plt.title(f"Percurso do TSP | Valor = {fobj} ({method})")
        plt.savefig('plot.png')
        plt.close()

    def calculate_distance_matrix(self):
        return np.sqrt(((self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]) ** 2).sum(axis=2))
        # distance_matrix = (distance_matrix * scale_factor).astype(int)

    # def calculate_total_distance(self):
    #     distance = 0
    #     for i in range(len(self.tour) - 1):
    #         distance += self.distance_matrix[self.tour[i]][self.tour[i + 1]]
    #     distance += self.distance_matrix[self.tour[-1]][self.tour[0]]
    #     return distance

    def __str__(self):
        tour_str = ' -> '.join(map(str, [t + 1 for t in self.tour]))  # Ajusta para a notação 1-indexed
        return (f"TSP Solution:\n"
                f"Name: {self.name}\n"
                f"Type: {self.type}\n"
                f"Dimension (Number of Cities): {self.dimension}\n"
                f"Tour: {tour_str}\n"
                f"Total Distance: {self.total_distance:.2f}"
                )
    
class TSP_Solution:

    def __init__(self, tsp, filename=None):
        self.tsp = tsp
        self.tour = np.arange(tsp.dimension)
        # np.random.shuffle(self.tour)
        self.objective = self.calculate_total_distance()
        if filename:
            self.read_file(filename)

    def read_file(self, filename):
        with open(filename, 'r') as f:
            self.tour = np.array(list(map(int, f.readline().split())))
            self.objective = float(f.readline())

    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write(' '.join(map(str, self.tour)) + '\n')
            f.write(f'{self.objective}\n')

    def calculate_total_distance(self):
        distance = 0
        for i in range(len(self.tour) - 1):
            distance += self.tsp.distance_matrix[self.tour[i], self.tour[i + 1]]
        # Retorna à cidade inicial
        distance += self.tsp.distance_matrix[self.tour[-1], self.tour[0]]
        return distance

    def evaluate(self):
        self.objective = self.calculate_total_distance()
        return self.objective

    def is_valid(self):
        if len(set(self.tour)) != len(self.tour):
            print('Tour inválido: cidades repetidas')
            return False
        if len(self.tour) != self.tsp.dimension:
            print('Tour inválido: número de cidades incorreto')
            return False
        return True

    # def reset(self):
    #     self.tour = np.arange(self.tsp.dimension)
    #     np.random.shuffle(self.tour)
    #     self.objective = self.calculate_total_distance()

    def copy_from(self, other):
        self.tour = other.tour.copy()
        self.objective = other.objective

    def swap(self, i, j):
        self.tour[i], self.tour[j] = self.tour[j], self.tour[i]
        self.objective = self.calculate_total_distance()  # Recalcula a distância após a troca

    def __str__(self):
        tour_str = ' -> '.join(map(str, self.tour)) + f' -> {self.tour[0]}'
        return f'Tour: {tour_str}\nTotal Distance: {self.objective:.2f}'

    def __eq__(self, other):
        return np.array_equal(self.tour, other.tour)

class ConstructionHeuristics:

    def __init__(self, tsp):
        self.tsp = tsp
        # Lista de cidades (índices) para embaralhar e gerar soluções aleatórias
        self.cities = np.arange(tsp.dimension)

    def random_solution(self):
        solution = TSP_Solution(self.tsp)
        np.random.shuffle(solution.tour)
        solution.evaluate()
        return solution

if __name__ == '__main__':
    tsp = TSP(filename='ALL_tsp/a280.tsp')
    sol = ConstructionHeuristics(tsp).random_solution()
    # sol.random_solution()
    tsp.tour = sol.tour
    tsp.plot_tsp('random',sol.objective)
    # print(sol.objective)
    # print(sol.calculate_total_distance())
    
    # tsp.plot_tsp()