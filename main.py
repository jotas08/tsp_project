import re
import numpy as np
import matplotlib.pyplot as plt

#Traveling Salesman Problem = Caixeiro Viajante


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
        self.total_distance = self.calculate_total_distance()

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
        plt.savefig(f'{method}.png')
        plt.close()

    def calculate_distance_matrix(self):
        return np.sqrt(((self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]) ** 2).sum(axis=2))
        # distance_matrix = (distance_matrix * scale_factor).astype(int)

    def calculate_total_distance(self):
        distance = 0
        for i in range(len(self.tour) - 1):
            distance += self.distance_matrix[self.tour[i]][self.tour[i + 1]]
        distance += self.distance_matrix[self.tour[-1]][self.tour[0]]
        return distance

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
        self.objective = self.calculate_total_distance()

    def __str__(self):
        tour_str = ' -> '.join(map(str, self.tour)) + f' -> {self.tour[0]}'
        return f'Tour: {tour_str}\nTotal Distance: {self.objective:.2f}'

    def __eq__(self, other):
        return np.array_equal(self.tour, other.tour)

class ConstructionHeuristics:

    def __init__(self, tsp):
        self.tsp = tsp
        self.cities = np.arange(tsp.dimension)

    def random_solution(self):
        solution = TSP_Solution(self.tsp)
        np.random.shuffle(solution.tour)
        solution.evaluate()
        return solution
    
    def greedy(self, start_city=0):
        solution = TSP_Solution(self.tsp)
        
        visited = np.zeros(self.tsp.dimension, dtype=bool)
        current_city = start_city
        solution.tour[0] = current_city
        visited[current_city] = True

        for i in range(1, self.tsp.dimension):
            nearest_city = None
            min_distance = np.inf
            for j in range(self.tsp.dimension):
                if not visited[j] and self.tsp.distance_matrix[current_city, j] < min_distance:
                    nearest_city = j
                    min_distance = self.tsp.distance_matrix[current_city, j]
            
            solution.tour[i] = nearest_city
            visited[nearest_city] = True
            current_city = nearest_city

        solution.evaluate()
        return solution

class LocalSearch:
    '''Local Search for the Traveling Salesman Problem (TSP)'''

    def __init__(self, tsp):
        self.tsp = tsp

    def two_opt(self, sol, first_improvement=True):
        ''' 
        Aplica a heurística 2-opt para melhorar o tour atual.
        
        Parameters:
            sol: TSP_Solution - a solução atual a ser melhorada
            first_improvement: bool (default True) - se True, para na primeira melhoria encontrada.
        
        Returns:
            bool - True se a solução foi melhorada, False caso contrário.
        '''
        best_distance = sol.objective
        tour = sol.tour
        improved = False

        # Tenta todas as combinações de pares de arestas (i, k)
        for i in range(1, len(tour) - 2):
            for k in range(i + 1, len(tour) - 1):
                # Calcula a mudança de distância ao trocar as arestas
                delta = self.calculate_2opt_gain(tour, i, k)

                # Se a mudança reduzir a distância total
                if delta < 0:
                    # Inverte a subsequência do tour entre i e k
                    tour[i:k+1] = list(reversed(tour[i:k+1]))
                    sol.evaluate()  # Recalcula a distância total
                    improved = True

                    if first_improvement:
                        return True  # Para na primeira melhoria se first_improvement for True

        return improved  # Retorna True se o tour foi melhorado

    def calculate_2opt_gain(self, tour, i, k):
        '''
        Calcula o ganho de distância que resultaria da inversão da subsequência entre i e k.
        
        Parameters:
            tour: lista de cidades no tour atual
            i, k: inteiros, índices das arestas a serem trocadas
        
        Returns:
            float - o ganho de distância (negativo se melhorar, positivo se piorar)
        '''
        # Acessa as cidades nos pontos i, k e nos pontos seguintes (para formar as arestas)
        city1, city2 = tour[i - 1], tour[i]
        city3, city4 = tour[k], tour[k + 1]

        # Calcula a diferença de distância ao trocar as arestas
        old_cost = self.tsp.distance_matrix[city1, city2] + self.tsp.distance_matrix[city3, city4]
        new_cost = self.tsp.distance_matrix[city1, city3] + self.tsp.distance_matrix[city2, city4]

        return new_cost - old_cost

# class Metaheuristics:



if __name__ == '__main__':
    tsp = TSP(filename='ALL_tsp/a280.tsp')
    # tsp.plot_tsp('initial_state', tsp.total_distance)
    sol = ConstructionHeuristics(tsp).greedy()
    tsp.tour = sol.tour
    tsp.plot_tsp('greedy',sol.objective)
    # sol = ConstructionHeuristics(tsp).random_solution()
    # tsp.tour = sol.tour
    # tsp.plot_tsp('random',sol.objective)
    local_search = LocalSearch(tsp)
    improved = local_search.two_opt(sol)  # Passa a solução aleatória para 2-opt

    # Atualiza o tour e plota a solução após o 2-opt
    if improved:
        tsp.tour = sol.tour  # Atualiza o tour no TSP com o novo tour melhorado
        tsp.plot_tsp('2opt_improved', sol.objective)  # Plota a solução melhorada
    else:
        print("Nenhuma melhoria foi encontrada com 2-opt.")