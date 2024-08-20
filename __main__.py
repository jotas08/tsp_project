import re
import numpy as np

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


if __name__ == '__main__':
    tsp = TSP(filename='ALL_tsp/kroE100.tsp')
    # print(tsp.coords[:10])