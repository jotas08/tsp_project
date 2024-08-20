import re
import numpy as np

#Usar Matplotlib e ReportLab para visualização do projeto

class TSP:

    def __init__(self, **args):
        if 'filename' in args:
            self.filename = args['filename']
            self.read_file(args['filename'])
        elif 'dimension' in args and 'coords' in args:
            self.dimension = args['dimension']
            self.coords = np.array(args['coords'])
        else:
            raise ValueError('Invalid arguments')

    def read_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        self.name = None
        self.type = None
        self.dimension = 0
        self.coords = []

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
                # Assume que esta é uma linha de coordenadas
                parts = line.split()
                if len(parts) == 3:
                    _, x, y = parts
                    self.coords.append((float(x), float(y)))

        self.coords = np.array(self.coords)

class TSPSolution:
    def __init__(self, tour, coords, objective=None):
        """
        :param tour: Lista de inteiros representando a sequência de índices das cidades no tour.
        :param coords: Array NumPy com as coordenadas das cidades.
        :param objective: (Opcional) Custo ou distância total do tour. Se não fornecido, será calculado.
        """
        self.tour = tour
        self.coords = coords
        self.objective = objective if objective is not None else self.calculate_total_distance()

    def calculate_total_distance(self):
        """
        Calcula a distância total do tour baseado nas coordenadas das cidades.
        """
        total_distance = 0
        num_cities = len(self.tour)
        for i in range(num_cities - 1):
            total_distance += np.linalg.norm(self.coords[self.tour[i]] - self.coords[self.tour[i + 1]])
        # Adiciona a distância para voltar ao ponto inicial
        total_distance += np.linalg.norm(self.coords[self.tour[-1]] - self.coords[self.tour[0]])
        return total_distance

    def write_file(self, filename):
        """
        Escreve a solução TSP em um arquivo.
        """
        with open(filename, 'w') as f:
            f.write('Tour:\n')
            for index in self.tour:
                f.write(f'{index + 1}: {self.coords[index][0]} {self.coords[index][1]}\n')
            f.write(f'\nDistância Total: {self.objective}\n')

    def __str__(self):
        """
        Representa a solução TSP como string.
        """
        s = 'Tour:\n'
        for index in self.tour:
            s += f'{index + 1}: {self.coords[index][0]} {self.coords[index][1]}\n'
        s += f'\nDistância Total: {self.objective}\n'
        return s

if __name__ == '__main__':
    tsp = TSP(filename='ALL_tsp/kroE100.tsp')
    # print(tsp.coords[:10])