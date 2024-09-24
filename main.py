import re
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq

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
        # distance_matrix = (distance_matrix * scale_factor).astype(int) CASO MELHORE TRANSFORMAR EM MATRIZ DE INTEIROS

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

    def reset(self):
        self.tour = np.arange(self.tsp.dimension)
        np.random.shuffle(self.tour)
        self.objective = self.calculate_total_distance()

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
    
    def find_nearest_city(self, current_city, visited):
        '''Encontra a cidade mais próxima que ainda não foi visitada.'''
        nearest_city = None
        nearest_distance = float('inf')

        for city in range(len(self.tsp.coords)):
            if not visited[city]:  # Considera apenas cidades não visitadas
                distance = self.tsp.distance_matrix[current_city][city]
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_city = city

        return nearest_city
    
    # def greedy(self, start_city=0):
    #     solution = TSP_Solution(self.tsp)
        
    #     visited = np.zeros(self.tsp.dimension, dtype=bool)
    #     current_city = start_city
    #     solution.tour[0] = current_city
    #     visited[current_city] = True

    #     for i in range(1, self.tsp.dimension):
    #         nearest_city = None
    #         min_distance = np.inf
    #         for j in range(self.tsp.dimension):
    #             if not visited[j] and self.tsp.distance_matrix[current_city, j] < min_distance:
    #                 nearest_city = j
    #                 min_distance = self.tsp.distance_matrix[current_city, j]
            
    #         solution.tour[i] = nearest_city
    #         visited[nearest_city] = True
    #         current_city = nearest_city

    #     solution.evaluate()
    #     return solution
    
    def greedy(self):
        '''Heurística gulosa para gerar uma solução inicial para o TSP.'''
        n = len(self.tsp.coords)  # Número de cidades
        visited = [False] * n  # Marca as cidades visitadas

        # Inicializa uma solução para o TSP
        solution = TSP_Solution(self.tsp)
        solution.tour = [-1] * n  # Inicializa o tour com valores inválidos

        # Começa na cidade 0
        current_city = 0
        visited[current_city] = True  # Marca a cidade como visitada
        solution.tour[0] = current_city  # Define a primeira cidade no tour

        for i in range(1, n):
            # Encontra a cidade mais próxima não visitada
            nearest_city = self.find_nearest_city(current_city, visited)

            # Verifica se uma cidade válida foi encontrada
            if nearest_city is None:
                raise ValueError(f"Nenhuma cidade válida encontrada na iteração {i}. Verifique a função find_nearest_city.")

            # Atualiza o tour com a cidade encontrada
            solution.tour[i] = nearest_city
            visited[nearest_city] = True  # Marca a cidade como visitada
            current_city = nearest_city  # Atualiza a cidade atual

        solution.evaluate()  # Avalia a distância total do tour
        return solution  # Retorna a solução gerada

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
                if delta < 0:
                    tour[i:k+1] = list(reversed(tour[i:k+1]))
                    sol.evaluate()  # Recalcula a distância total
                    improved = True

                    if first_improvement:
                        return True  # Para na primeira melhoria se first_improvement for True

        return improved  # Retorna True se o tour foi melhorado
    
    def calculate_2opt_gain(self, tour, i, k):
        city1, city2 = tour[i - 1], tour[i]
        city3, city4 = tour[k], tour[k + 1]

        old_cost = self.tsp.distance_matrix[city1, city2] + self.tsp.distance_matrix[city3, city4]
        new_cost = self.tsp.distance_matrix[city1, city3] + self.tsp.distance_matrix[city2, city4]

        return new_cost - old_cost
    
    def swap(self, sol, first_improvement=True):
        '''Tenta melhorar a solução trocando duas cidades no tour.'''
        tour = sol.tour
        improved = False

        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                # Calcula a mudança de distância ao trocar duas cidades
                old_distance = self.calculate_swap_cost(tour, i, j)
                tour[i], tour[j] = tour[j], tour[i]  # Troca as cidades
                sol.evaluate()  # Recalcula a distância após a troca

                if sol.objective < old_distance:
                    improved = True
                    if first_improvement:
                        return True  # Para na primeira melhoria
                else:
                    # Desfaz a troca se não houver melhoria
                    tour[i], tour[j] = tour[j], tour[i]
        return improved
    
    def calculate_swap_cost(self, tour, i, j):
        city_before_i = tour[i - 1] if i > 0 else tour[-1]
        city_after_i = tour[i + 1] if i < len(tour) - 1 else tour[0]
        city_before_j = tour[j - 1] if j > 0 else tour[-1]
        city_after_j = tour[j + 1] if j < len(tour) - 1 else tour[0]

        # Cálculo das distâncias atuais
        current_cost = (self.tsp.distance_matrix[city_before_i, tour[i]] + self.tsp.distance_matrix[tour[i], city_after_i] +
                        self.tsp.distance_matrix[city_before_j, tour[j]] + self.tsp.distance_matrix[tour[j], city_after_j])

        # Cálculo das distâncias após a troca
        new_cost = (self.tsp.distance_matrix[city_before_i, tour[j]] + self.tsp.distance_matrix[tour[j], city_after_i] +
                    self.tsp.distance_matrix[city_before_j, tour[i]] + self.tsp.distance_matrix[tour[i], city_after_j])

        return new_cost - current_cost
    
    def VND(self, sol, first_improvement=True):
        '''Variable Neighborhood Descent para o TSP combinando two_opt e swap.'''
        any_improved = False

        while True:
            improved = False
            # Aplica 2-opt e depois swap, reiniciando quando uma melhoria for encontrada
            if self.two_opt(sol, first_improvement):
                improved = True
            if self.swap(sol, first_improvement):
                improved = True

            if not improved:
                break  # Para quando nenhuma melhoria é encontrada
            any_improved = True

        return any_improved

class Metaheuristics:
    def __init__(self):
        pass

    @staticmethod
    def RMS(tsp, max_tries=1000, first_imp=True,timeout=None):
        '''Randomized Multi-Start, uma metaheurística que combina heurísticas de construção e métodos de busca local.
        
        Parameters:
            tsp: TSP - o problema do TSP a ser resolvido
            max_tries: int (default 1000) - máximo de tentativas sem melhoria
            first_imp: bool (default True) - se True, a busca local usará a primeira melhoria encontrada
        
        Returns:
            TSP_Solution - a melhor solução encontrada
        '''
        # Inicializa as heurísticas de construção e busca local
        ch = ConstructionHeuristics(tsp)
        ls = LocalSearch(tsp)
        
        best = None  # Inicializa a melhor solução como None
        ite = 0  # Iterador para contar as tentativas
        start_time = time.time()
        
        while ite < max_tries:

            if timeout and (time.time() - start_time) > timeout:
                print("Timeout atingido.")
                break

            ite += 1
            # Gera uma solução aleatória com o construtor
            sol = ch.random_solution()
            
            # Aplica a busca local VND
            ls.VND(sol, first_imp)
            
            # Verifica se a nova solução é melhor que a melhor encontrada até agora
            if not best or sol.objective + 1e-6 < best.objective:
                best = sol  # Atualiza a melhor solução
                ite = 0  # Reinicia o contador de iterações sem melhoria
                if __debug__:
                    print('RMS:', best.objective)
        
        return best
    
    @staticmethod
    def perturbation(sol, k=2):
        '''Perturba o tour atual trocando k pares de cidades aleatoriamente no tour.
        Parameters:
            sol: TSP_Solution - a solução atual
            k: int (default 2) - número de pares de cidades a serem trocados (perturbação)
        '''
        tour = sol.tour
        n = len(tour)

        for _ in range(k):
            # Escolhe dois índices aleatórios no tour para realizar a troca
            i, j = np.random.choice(n, size=2, replace=False)
            # Troca as cidades
            tour[i], tour[j] = tour[j], tour[i]

        sol.evaluate()  # Recalcula o valor objetivo (distância total) após a perturbação

    @staticmethod
    def ILS(tsp, max_tries=1000, first_imp=True, k=2):
        '''Iterated Local Search para o TSP.
        
        Parameters:
            tsp: TSP - o problema do TSP a ser resolvido
            max_tries: int (default 1000) - máximo de tentativas sem melhoria
            first_imp: bool (default True) - se True, a busca local usará a primeira melhoria encontrada
            k: int (default 2) - número de perturbações (quantas trocas de cidades fazer durante a perturbação)
        
        Returns:
            TSP_Solution - a melhor solução encontrada
        '''
        ch = ConstructionHeuristics(tsp)
        ls = LocalSearch(tsp)
        
        # Gera uma solução inicial gulosa
        best = ch.greedy()
        ls.VND(best,first_imp)  # Aplica busca local VND na solução inicial
        if __debug__:
            print('ILS inicial:', best.objective)

        # Copia a solução inicial para a solução de trabalho
        sol = TSP_Solution(tsp)
        sol.copy_from(best)
        
        ite = 0  # Contador de iterações sem melhoria
        while ite < max_tries:
            ite += 1

            # Perturba a solução de trabalho
            Metaheuristics.perturbation(sol, k=k)
            
            # Aplica busca local após a perturbação
            ls.VND(sol, first_imp)
            
            # Se a solução perturbada é melhor que a melhor encontrada, atualiza
            if sol.objective + 1e-6 < best.objective:
                best.copy_from(sol)
                ite = 0  # Reseta o contador de iterações sem melhoria
                if __debug__:
                    print('ILS melhorado:', best.objective)

        return best

    @staticmethod
    def VNS(tsp, max_tries=1000, first_imp=True):
        '''Variable Neighborhood Search para o TSP.
        
        Parameters:
            tsp: TSP - o problema do TSP a ser resolvido
            max_tries: int (default 1000) - máximo de tentativas sem melhoria
            first_imp: bool (default True) - se True, a busca local usará a primeira melhoria encontrada
        
        Returns:
            TSP_Solution - a melhor solução encontrada
        '''
        ch = ConstructionHeuristics(tsp)
        ls = LocalSearch(tsp)

        # Gera uma solução inicial gulosa
        best = ch.greedy()
        ls.VND(best, first_imp)  # Aplica busca local VND na solução inicial
        sol = TSP_Solution(tsp)  # Solução de trabalho
        sol.copy_from(best)

        if __debug__:
            print('VNS inicial:', best.objective)

        ite = 0  # Contador de iterações sem melhoria
        k = 1  # Começa com uma pequena perturbação
        k_max = len(best.tour) // 2  # Define o k_max como metade do número de cidades

        while ite < max_tries:
            ite += 1
            last_objective = sol.objective

            # Perturba a solução de trabalho
            Metaheuristics.perturbation(sol, k=k)
            
            # Aplica busca local após a perturbação
            ls.VND(sol, first_imp)

            # Ajusta a intensidade da perturbação (k)
            if np.isclose(sol.objective, last_objective):
                k = k + 1 if k < k_max else 1  # Aumenta k se não houve melhoria
            else:
                k = k - 1 if k > 1 else 1  # Reduz k se houve melhoria

            # Verifica se a nova solução é melhor que a melhor solução encontrada até agora
            if sol.objective + 1e-6 < best.objective:
                best.copy_from(sol)  # Atualiza a melhor solução
                ite = 0  # Reinicia o contador de tentativas sem melhoria
                if __debug__:
                    print('VNS melhorado:', best.objective)

        return best

    @staticmethod
    def Tabu(tsp, max_tries=1000, first_imp=True, tenure=10):
        '''Tabu Search para o TSP.
        
        Parameters:
            tsp: TSP - o problema do TSP a ser resolvido
            max_tries: int (default 1000) - número máximo de tentativas sem melhoria
            first_imp: bool (default True) - se True, a busca local usará a primeira melhoria encontrada
            tenure: int (default 10) - número de iterações que uma regra é considerada tabu
        
        Returns:
            TSP_Solution - a melhor solução encontrada
        '''
        tabu_list = deque(maxlen=tenure)  # Lista tabu com capacidade limitada pelo tenure

        def add_tabu_move(i, j):
            '''Adiciona uma troca entre as cidades i e j na lista tabu.'''
            tabu_list.append((i, j))  # Adiciona o movimento (troca) à lista tabu

        def is_tabu(i, j):
            '''Verifica se a troca entre as cidades i e j está na lista tabu.'''
            return (i, j) in tabu_list or (j, i) in tabu_list

        ch = ConstructionHeuristics(tsp)
        ls = LocalSearch(tsp)

        # Gera uma solução inicial usando a heurística gulosa
        best = ch.greedy()
        ls.VND(best, first_imp)  # Aplica a busca local VND na solução inicial

        if __debug__:
            print('Tabu Search inicial:', best.objective)

        sol = TSP_Solution(tsp)  # Solução de trabalho
        sol.copy_from(best)

        ite = 0
        while ite < max_tries:
            ite += 1
            best_objective_before_tabu = sol.objective

            # Realiza uma perturbação (swap) e adiciona o movimento à lista tabu
            city1, city2 = None, None
            for i in range(len(sol.tour)):
                for j in range(i + 1, len(sol.tour)):
                    # Tenta realizar uma troca entre as cidades i e j que não esteja na lista tabu
                    if not is_tabu(i, j):
                        # Salva as cidades que serão trocadas
                        city1, city2 = i, j
                        break
                if city1 is not None:
                    break

            if city1 is None or city2 is None:
                break  # Não há mais trocas possíveis

            # Realiza a troca entre city1 e city2
            sol.tour[city1], sol.tour[city2] = sol.tour[city2], sol.tour[city1]
            sol.evaluate()  # Recalcula o valor objetivo do tour

            # Adiciona a troca à lista tabu
            add_tabu_move(city1, city2)

            # Aplica busca local (VND) após a perturbação
            ls.VND(sol, first_imp)

            # Critério de aspiração: se a nova solução for melhor, ignoramos a tabu list
            if sol.objective + 1e-6 < best.objective:
                best.copy_from(sol)
                ite = 0  # Reinicia o contador de tentativas sem melhoria
                if __debug__:
                    print('Tabu Search melhorado:', best.objective)
            else:
                # Se a solução não melhorou significativamente, desfaz a troca (critério de aspiração)
                sol.tour[city1], sol.tour[city2] = sol.tour[city2], sol.tour[city1]
                sol.evaluate()

            # Se a nova solução é muito pior (mais de 5% pior), resetamos a tabu list
            if sol.objective > best_objective_before_tabu * 1.05:
                tabu_list.clear()  # Reseta a lista tabu
                ls.VND(sol, first_imp)  # Aplica VND novamente após reset

        return best  # Retorna a melhor solução encontrada

    @staticmethod
    def SA(tsp, max_tries=1000, first_imp=True, T0=1e3, alpha=0.99):
        '''Simulated Annealing para o TSP.
        
        Parameters:
            tsp: TSP - o problema do TSP a ser resolvido
            max_tries: int (default 1000) - número máximo de tentativas sem melhoria
            first_imp: bool (default True) - se True, a busca local usará a primeira melhoria encontrada
            T0: float (default 1e3) - temperatura inicial
            alpha: float (default 0.99) - fator de resfriamento (cooling factor)
        
        Returns:
            TSP_Solution - a melhor solução encontrada
        '''
        # Heurística de construção: gera uma solução inicial gulosa
        ch = ConstructionHeuristics(tsp)
        best = ch.greedy()  # Gera a solução inicial
        ls = LocalSearch(tsp)

        # Aplica a busca local (VND) na solução inicial
        ls.VND(best, first_imp)
        current = TSP_Solution(tsp)
        current.copy_from(best)  # A solução atual começa como a melhor solução
        sol = TSP_Solution(tsp)  # Solução temporária usada para as perturbações

        ite = 0
        print('SA inicial:', best.objective)

        # Simulated Annealing Loop
        while ite < max_tries:
            ite += 1

            # Copia a solução atual para a temporária
            sol.copy_from(current)

            # Perturbação: troca 2 cidades aleatórias no tour
            i, j = np.random.choice(len(sol.tour), size=2, replace=False)
            sol.tour[i], sol.tour[j] = sol.tour[j], sol.tour[i]
            sol.evaluate()  # Recalcula a distância total do tour

            # Calcula a diferença entre a nova solução e a atual
            delta = (sol.objective - current.objective) / best.objective

            # Aceita a nova solução se for melhor ou com uma certa probabilidade
            if delta < -1e-3 or np.random.rand() < np.exp(-delta / T0):
                current.copy_from(sol)  # Aceita a nova solução

                # Se a nova solução for melhor que a melhor solução encontrada, atualiza
                if current.objective + 1e-6 < best.objective:
                    best.copy_from(current)
                    ite = 0  # Reinicia o contador de tentativas sem melhoria
                    if __debug__:
                        print('SA melhorado:', best.objective)

            # Resfriamento: reduz a temperatura
            T0 *= alpha

        return best  # Retorna a melhor solução encontrada

    @staticmethod
    def GRASP(tsp, max_tries=1000, first_imp=True, K=10):
        '''Greedy Randomized Adaptive Search Procedure para o TSP.
        
        Parameters:
            tsp: TSP - o problema do TSP a ser resolvido
            max_tries: int (default 1000) - número máximo de tentativas sem melhoria
            first_imp: bool (default True) - se True, a busca local usará a primeira melhoria encontrada
            K: int (default 10) - número de candidatos a serem considerados na construção gulosa randomizada
        
        Returns:
            TSP_Solution - a melhor solução encontrada
        '''

        # Função que realiza a construção gulosa randomizada
        def greedy_randomized(solution):
            n = len(tsp.coords)  # Número de cidades
            visited = [False] * n  # Marca as cidades visitadas
            solution.reset()  # Reinicializa a solução

            # Começa na cidade 0
            current_city = 0
            visited[current_city] = True
            solution.tour[0] = current_city  # Define a primeira cidade no tour

            # Itera para construir o tour
            for i in range(1, n):
                candidates = []

                # Para cada cidade não visitada, calcula o custo de inserção
                for j in range(n):
                    if not visited[j]:
                        # Custo da cidade j
                        cost = tsp.distance_matrix[current_city][j]
                        heapq.heappush(candidates, (cost, j))

                        # Mantém apenas os K melhores candidatos
                        if len(candidates) > K:
                            heapq.heappop(candidates)

                # Escolhe aleatoriamente um dos K melhores candidatos
                _, next_city = candidates[np.random.randint(len(candidates))]

                # Atualiza o tour com a cidade selecionada
                solution.tour[i] = next_city
                visited[next_city] = True  # Marca como visitada
                current_city = next_city  # Atualiza a cidade atual

            solution.evaluate()  # Avalia a solução gerada

        # GRASP main loop
        ch = ConstructionHeuristics(tsp)
        best = ch.greedy()  # Gera uma solução inicial gulosa
        ls = LocalSearch(tsp)
        ls.VND(best, first_imp)  # Aplica a busca local na solução inicial

        sol = TSP_Solution(tsp)  # Solução de trabalho

        ite = 0
        while ite < max_tries:
            ite += 1

            # Gera uma solução gulosa randomizada
            greedy_randomized(sol)

            # Aplica busca local à solução gerada
            ls.VND(sol, first_imp)

            # Se a nova solução for melhor que a melhor solução encontrada, atualiza
            if sol.objective + 1e-6 < best.objective:
                best.copy_from(sol)
                ite = 0  # Reinicia o contador de tentativas sem melhoria

                if __debug__:
                    print('GRASP melhorado:', best.objective)

        return best  # Retorna a melhor solução encontrada

# class MIP:
#p-meta

if __name__ == '__main__':
    ini = time.time()
    tsp = TSP(filename='ALL_tsp/att48.tsp')
    # tsp.plot_tsp('initial_state', tsp.total_distance)
    # sol = ConstructionHeuristics(tsp).greedy()
    # tsp.tour = sol.tour
    # tsp.plot_tsp('greedy',sol.objective)
    # sol = ConstructionHeuristics(tsp).random_solution()
    # tsp.tour = sol.tour
    # tsp.plot_tsp('random',sol.objective)
    # local_search = LocalSearch(tsp)
    # improved = local_search.VND(sol)

    # if improved:
    #     tsp.tour = sol.tour
    #     tsp.plot_tsp('VND-random', sol.objective)
    # else:
    #     print("Nenhuma melhoria foi encontrada com 2-opt.")
    # best_sol = Metaheuristics.RMS(tsp,max_tries=1000,first_imp=True,timeout=30)
    # best_sol = Metaheuristics.ILS(tsp,max_tries=50,first_imp=True, k=2)
    # best_sol = Metaheuristics.VNS(tsp, max_tries=100, first_imp=True)
    # best_sol = Metaheuristics.Tabu(tsp, max_tries=1000, first_imp=True, tenure=10)
    # best_sol = Metaheuristics.SA(tsp, max_tries=1000, first_imp=True, T0=1e3, alpha=0.99)
    best_sol = Metaheuristics.GRASP(tsp, max_tries=200, first_imp=True, K=10)

    tsp.tour = best_sol.tour
    tsp.plot_tsp('GRASP',best_sol.objective)
    print('Tempo total = ',time.time() - ini,"\n",best_sol)