import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import permutations
from tqdm import tqdm


def get_all(matriz, tipo):
    res = []

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):

            if matriz[i, j] == tipo:
                res.append([i, j])

    return res


def select_position(coords, pos):
    row = coords[0]
    col = coords[1]

    if pos == 1:
        new_row = row
        new_col = col - 1

    elif pos == 2:
        new_row = row - 1
        new_col = col

    elif pos == 3:
        new_row = row
        new_col = col + 1

    elif pos == 4:
        new_row = row + 1
        new_col = col

    elif pos == 5:
        new_row = row + 1
        new_col = col - 1

    elif pos == 6:
        new_row = row - 1
        new_col = col - 1

    elif pos == 7:
        new_row = row - 1
        new_col = col + 1

    elif pos == 8:
        new_row = row + 1
        new_col = col + 1

    elif pos == 9:
        new_row = row
        new_col = col - 2

    elif pos == 10:
        new_row = row - 2
        new_col = col

    elif pos == 11:
        new_row = row
        new_col = col + 2

    elif pos == 12:
        new_row = row + 2
        new_col = col

    else:
        print("No se reconoce el movimiento")

    if new_row < 0 or new_col < 0:
        return None
    else:
        return [new_row, new_col]


def get_value(matrix, coords):
    if coords is None or coords[0] >= matrix.shape[0] or coords[1] >= matrix.shape[1]:
        return None
    else:
        return matrix[coords[0], coords[1]]


def change_val(matrix, coords, val):
    matrix[coords[0], coords[1]] = val

    return matrix


def get_surroundings(matrix, coords):
    surroundings = []

    for i in range(1, 5):
        c = select_position(coords, i)

        if c is not None:
            val = get_value(matrix, c)
            surroundings.append([c, val])

    return surroundings


def get_extended_surroundings(matrix, coords):
    surroundings = []

    for i in range(1, 9):
        c = select_position(coords, i)

        if c is not None:
            val = get_value(matrix, c)
            surroundings.append([c, val])

    return surroundings


def get_diagonales(matrix, coords):
    surroundings = []

    for i in [6, 7, 5, 8]:
        c = select_position(coords, i)

        if c is not None:
            val = get_value(matrix, c)
            surroundings.append([c, val])

    return surroundings


def swap(matrix, coord_1, coord_2):
    val_1 = get_value(matrix, coord_1)
    val_2 = get_value(matrix, coord_2)

    matrix[coord_1[0], coord_1[1]] = val_2
    matrix[coord_2[0], coord_2[1]] = val_1

    return matrix


def get_offset(current_pos, new_pos):
    return list((np.array(new_pos) - np.array(current_pos)))



class Autopoiesis():

    def __init__(self, matrix_size, tolerance, weights=None, particles=["h", "k", "s", "l"]):

        self.rows = matrix_size[0]
        self.cols = matrix_size[1]
        self.tolerance = tolerance
        self.weights = weights

        self.matrix = None
        self.current_bls = {}
        self.particles = particles

    def inicializar(self):

        if self.weights is not None:

            p_weights = [self.weights[p] for p in self.particles]
            self.matrix = np.random.choice(self.particles,
                                           size=(self.rows, self.cols),
                                           p=p_weights)

            self.matrix = self.matrix.astype("<U2")

        else:

            self.matrix = np.random.choice(self.particles,
                                           size=(self.rows, self.cols))

            self.matrix = self.matrix.astype("<U2")

    def desplazamiento_L(self, pos_L):

        # Función para definir el movimiento de una partícula L
        # El output es la matriz actualizada y la coordenada con la que la posicion original debe swapear

        # Primero obtenemos los surroundings de L
        surroundings = get_surroundings(self.matrix, pos_L)

        surrounding_holes = list(filter(lambda x: x[1] == "h", surroundings))
        surrounding_s = list(filter(lambda x: x[1] == "s", surroundings))

        # Chequeamos si hay algún hole alrededor de pos_L
        if surrounding_holes != []:

            # seleccionamos uno al azar y swap
            ix = np.random.randint(0, len(surrounding_holes))

            # Pasamos L al hole
            self.matrix = swap(self.matrix, pos_L, surrounding_holes[ix][0])

            # Retornamos la matriz y la coordenada con que debe swapear el pos original
            return pos_L, surrounding_holes[ix][0]

        # Si hay un S alrededor
        elif surrounding_s != []:

            # Seleccionamos uno al azar
            ix = np.random.randint(0, len(surrounding_s))

            # Nos quedamos con sus coordenadas
            target_S = surrounding_s[ix][0]

            # Ejecutamos las reglas de desplazamiento de S
            # Si se encuentra una posición a la que S pueda moverse, se ejecutará el swap y se entregará
            # la matriz actualizada junto con la posición con la que debe swapear a L
            pos_S = self.desplazamiento_S(target_S)
            self.matrix = swap(self.matrix, pos_L, pos_S)

            return pos_L, pos_S

        # Si no hay holes ni S alrededor de pos_L
        else:

            # Retornamos el mismo pos_L para que swapee con current_cor
            return pos_L, pos_L

    def desplazamiento_S(self, pos_S):

        # Función para definir el movimiento de una partícula S
        # El output es la matriz actualizada y la coordenada con la que la posicion original debe swapear

        # Primero obtenemos los surroundings de S
        surroundings = get_surroundings(self.matrix, pos_S)

        surrounding_holes = list(filter(lambda x: x[1] == "h", surroundings))
        surrounding_bl = list(filter(lambda x: x[1] == "bl", surroundings))

        # Chequeamos si hay algún hole alrededor de pos_S
        if surrounding_holes != []:

            # seleccionamos uno al azar y swap
            ix = np.random.randint(0, len(surrounding_holes))

            # Pasamos S al hole
            self.matrix = swap(self.matrix, pos_S, surrounding_holes[ix][0])

            # Retornamos la matriz y la coordenada con que debe swapear el pos original

            return pos_S

        # Si no hay holes alrededor pero hay BL...
        elif surrounding_bl != []:

            # Si tenemos más de un bl alrededor del S
            if len(surrounding_bl) > 1:

                possible_holes = []

                # Primero chequeamos si existen 2-step holes
                for i in surrounding_bl:

                    # Para cada BL, offseteamos en la direccion correspondiente
                    direccion = np.array(get_offset(pos_S, i[0])) * 2
                    temp = np.array(pos_S) + direccion

                    # Si el valor de esa posicion es "h", lo apendeamos a los h posibles
                    if get_value(self.matrix, temp) == "h":
                        possible_holes.append(temp)

                # Después calculamos los holes diagonales
                for i in permutations(surrounding_bl, 2):

                    # Para cada par de BL calculamos el offset respecto a la posición de S
                    offset_1 = get_offset(pos_S, i[0][0])
                    offset_2 = get_offset(pos_S, i[1][0])

                    # Lo sumamos a la posición de S para generar las coordenadas
                    full_offset = np.array(offset_1) + np.array(offset_2)
                    diag_pos = np.array(pos_S) + full_offset

                    # Si la posición diagonal es igual a "h", lo apendeamos
                    if get_value(self.matrix, diag_pos) == "h":
                        possible_holes.append(diag_pos)

                if len(possible_holes) > 0:
                    # Cuando tenemos todos los posibles "h" calculados, seleccionamos uno al azar y swap
                    target_hole = possible_holes[np.random.randint(0, len(possible_holes))]

                    # Swapeamos S por el hole
                    self.matrix = swap(self.matrix, target_hole, pos_S)

                    # Retornamos la matriz actualizada y la posición por la que debe swapearse current pos

                return pos_S

            # Si hay solo un BL alrededor de pos_S, solo hay que testear la presencia de un "h" como 2-step
            else:

                # Offseteamos pos_S en la dirección necesaria
                direccion = np.array(get_offset(pos_S, surrounding_bl[0][0])) * 2
                posible = np.array(pos_S) + direccion

                # Evaluamos si hay un hole
                if get_value(self.matrix, posible) == "h":
                    self.matrix = swap(self.matrix, pos_S, posible)

                    return pos_S

                return pos_S

        # Si no hay holes ni BL alrededor de pos_S
        else:

            # Retornamos el mismo pos_S para que swapee con current_cor

            return pos_S

    def motion_H(self):

        # Juntamos todas las coordenadas de H
        coords = get_all(self.matrix, "h")

        # Permutamos
        random.shuffle(coords)

        # Para cada coordenada...
        for c in coords:

            # Seleccionamos una posición aleatoria
            neighbor_coord = select_position(c, np.random.randint(1, 4))
            val = get_value(self.matrix, neighbor_coord)

            if val is not None:

                # Si la partícula es S, L o K, swap
                if val in ["s", "k"]:

                    self.matrix = swap(self.matrix, c, neighbor_coord)

                # Si es un bl
                elif val == "bl":

                    offset = np.array(get_offset(neighbor_coord, c)) * 2
                    two_step = offset + np.array(c)

                    if get_value(self.matrix, two_step) == "s":
                        self.matrix = swap(self.matrix, two_step, c)

                elif val == "l":

                    self.matrix = swap(self.matrix, c, neighbor_coord)
                    self.bonding(c)

                else:
                    pass

    def motion_L(self):

        # Juntamos todas las coordenadas de L
        coords = get_all(self.matrix, "l")

        # Permutamos
        random.shuffle(coords)

        # Para cada coordenada...
        for c in coords:

            if get_value(self.matrix, c) == "l":

                # Seleccionamos una posición aleatoria
                neighbor_coord = select_position(c, np.random.randint(1, 4))
                val = get_value(self.matrix, neighbor_coord)

                if val is not None:

                    if val == "s":

                        to_swap = self.desplazamiento_S(neighbor_coord)
                        self.matrix = swap(self.matrix, c, to_swap)

                        self.bonding(to_swap)

                    elif val == "h":

                        self.matrix = swap(self.matrix, c, neighbor_coord)
                        self.bonding(neighbor_coord)

                    else:
                        pass

    def motion_K(self):

        # Juntamos todas las coordenadas de H
        coords = get_all(self.matrix, "k")

        # Permutamos
        random.shuffle(coords)

        # Para cada coordenada...
        for c in coords:

            if get_value(self.matrix, c) == "k":

                # Seleccionamos una posición aleatoria
                neighbor_coord = select_position(c, np.random.randint(1, 4))
                val = get_value(self.matrix, neighbor_coord)

                if val is not None:

                    if val == "s":

                        to_swap = self.desplazamiento_S(neighbor_coord)
                        self.matrix = swap(self.matrix, to_swap, c)

                    elif val == "l":

                        to_swap, pos_L = self.desplazamiento_L(neighbor_coord)
                        self.matrix = swap(self.matrix, to_swap, c)

                        self.bonding(pos_L)

                    elif val == "h":

                        self.matrix = swap(self.matrix, c, neighbor_coord)

                    else:
                        pass

    def get_reaction_pairs(self, coord_k):

        master = {}

        for i in range(1, 9):
            pos = select_position(coord_k, i)

            if get_value(self.matrix, pos) == "s":
                master[i] = pos

        pair_options = [(1, 6), (6, 2), (2, 7),
                        (7, 3), (3, 8), (8, 4),
                        (4, 5), (5, 1)]

        reaction_options = []

        for pair in pair_options:

            if pair[0] in master.keys() and pair[1] in master.keys():
                reaction_options.append([master[pair[0]], master[pair[1]]])

        return reaction_options

    def composition(self):

        coords = get_all(self.matrix, "k")

        for c in coords:

            reaction_pairs = self.get_reaction_pairs(c)

            if len(reaction_pairs) > 0:
                ix = np.random.randint(0, len(reaction_pairs))
                target = reaction_pairs[ix]

                random.shuffle(target)

                self.matrix = change_val(self.matrix, target[0], "l")
                self.matrix = change_val(self.matrix, target[1], "h")

    def desintegrar_base(self, coord):

        surroundings = get_extended_surroundings(self.matrix, coord)
        surr_holes = list(filter(lambda x: x[1] == "h", surroundings))

        if len(surr_holes) > 0:

            ix = np.random.randint(0, len(surr_holes))

            self.matrix = change_val(self.matrix, surr_holes[ix][0], "s")
            self.matrix = change_val(self.matrix, coord, "s")

        # QUÉ HACEMOS SI NO HAY HOLES?
        # Dejamos el valor porque no están dadas las condiciones para la desintegración

        else:

            self.matrix = change_val(self.matrix, coord, "s")

            full_holes = get_all(self.matrix, "h")

            ix = np.random.randint(0, len(full_holes))
            self.matrix = change_val(self.matrix, full_holes[ix], "s")

    def desintegrar_high(self, coord):

        # Sampleamos una variable para ver si es inferior a la tolerancia
        if np.random.random() > self.tolerance:

            if get_value(self.matrix, coord) == "l":

                self.desintegrar_base(coord)

            else:

                # REVISAMOS EL BOND

                # Si tiene un solo bond
                if self.check_if_single_bonded(coord):

                    # Chequeamos el estatus de su bond
                    bond = self.current_bls.get(f"COORD_{coord}")[0]

                    # Si su bond es single (o sea, solo estaba enlazado al target)
                    if self.check_if_single_bonded(bond):

                        # Lo reducimos a L
                        self.matrix = change_val(self.matrix, bond, "l")

                        # Lo removemos por completo de current_bls
                        self.current_bls.pop(f"COORD_{bond}")

                        # Removemos nuestro target de current_bls
                        self.current_bls.pop(f"COORD_{coord}")

                        # Lo desintegramos
                        self.desintegrar_base(coord)



                    # Si es double bonded
                    else:

                        # El bond no se ve afectado, solo alteramos el target

                        # Removemos nuestro target de current_bls
                        self.current_bls.pop(f"COORD_{coord}")

                        self.current_bls[f"COORD_{bond}"] = [x for x in self.current_bls.get(f"COORD_{bond}") if
                                                             x != coord]

                        # Lo desintegramos
                        self.desintegrar_base(coord)


                # Si el bl target es double bonded
                else:

                    # Separamos las coordenadas de sus bonds
                    target_bonds = self.current_bls.get(f"COORD_{coord}")

                    # Para cada uno
                    for i in target_bonds:

                        # Si es un single bond (o sea, solo estaba conectado a nuestro target)
                        if self.check_if_single_bonded(i):

                            # Lo reducimos a L
                            self.matrix = change_val(self.matrix, i, "l")

                            # Lo removemos por completo de current_bls
                            self.current_bls.pop(f"COORD_{i}")

                        # Si está double bonded
                        else:

                            # Removemos el target de los bonds del bond (pasa a ser single bonded)
                            self.current_bls[f"COORD_{i}"] = [x for x in self.current_bls.get(f"COORD_{i}") if
                                                              x != coord]

                    # Removemos nuestro target de current_bls
                    self.current_bls.pop(f"COORD_{coord}")

                    # Lo desintegramos
                    self.desintegrar_base(coord)

    def desintegrar(self):

        coords = get_all(self.matrix, "l")
        temp = get_all(self.matrix, "bl")

        coords.extend(temp)

        for c in coords:
            self.desintegrar_high(c)

    def check_if_single_bonded(self, coord):

        if f"COORD_{coord}" in self.current_bls.keys():
            if len(self.current_bls.get(f"COORD_{coord}")) == 1:
                return True
            else:
                return False

    def bonding(self, coord):

        if coord is not None and get_value(self.matrix, coord) == "l":

            # Examinamos todo lo que hay en los extended surroundings
            ext_surroundings = get_extended_surroundings(self.matrix, coord)

            # También en los surroundings por las dudas
            surroundings = get_surroundings(self.matrix, coord)

            # Extraemos las diagonales
            # diag_sorroundings = get_diagonales(self.matrix, coord)

            # Nos quedamos solo con los links y bonded links
            surr_l = list(filter(lambda x: x[1] == "l", ext_surroundings))
            surr_bl = list(filter(lambda x: x[1] == "bl", ext_surroundings))

            # Conservamos solo los bonded links que sean single bonded
            single_bl = [x for x in surr_bl if self.check_if_single_bonded(x[0])]

            # Conservamos solamente los que generen un ángulo mayor a 90°
            coord_surr = [x[0] for x in surroundings]
            coord_ext_surr = [x[0] for x in ext_surroundings]

            # available_bl va a contener las coordenadas que respeten esta condición
            available_bl = []

            for i in single_bl:

                # Conseguimos las coordenadas del sbl y su bond
                target_sbl = i[0]
                target_bond = self.current_bls.get(f"COORD_{target_sbl}")[0]

                # Lo único que necesitamos saber es si el bond de nuestro target está o no
                # dentro de los extended surroundings. Si ambos están en ext surr, todo enlace
                # va a ser de 45°
                if target_bond not in coord_ext_surr:
                    available_bl.append(target_sbl)

            # Si queda más de un bl en los alrededores
            if len(available_bl) > 1:

                # Generamos dos números aleatorios
                ix_to_bond = np.random.choice(list(range(len(available_bl))), 2, False)

                # Agregamos las coordenadas de nuestro link al diccionario con sus dos bl
                self.current_bls[f"COORD_{coord}"] = [available_bl[ix_to_bond[0]], available_bl[ix_to_bond[1]]]

                # Agregamos nuestro link como bond a los bl relacionados
                self.current_bls[f"COORD_{available_bl[ix_to_bond[0]]}"].append(coord)
                self.current_bls[f"COORD_{available_bl[ix_to_bond[1]]}"].append(coord)

                # Cambiamos el valor de nuestro link a bl
                self.matrix = change_val(self.matrix, coord, "bl")

                # return matrix, current_bls

            # Si queda solo un bl que cumpla las condiciones
            elif len(available_bl) == 1:

                # Agregamos nuestro link al diccionario con su bl
                self.current_bls[f"COORD_{coord}"] = [available_bl[0]]

                # Agregamos nuestro link como bond al bl relacionado
                self.current_bls[f"COORD_{available_bl[0]}"].append(coord)

                self.matrix = change_val(self.matrix, coord, "bl")

                # Todavía nos queda una posibilidad de enlace --> Revisamos los free links

                # Si hay algún free link en los alrededores
                if len(surr_l) > 0:

                    # available_l va a contener los free l que respeten la condición de ángulo
                    available_l = []

                    # Para cada free l
                    for i in surr_l:

                        # Chequeamos si el offset entre el bl con el que enlazamos y el free l es mayor a 1
                        # Si esto es verdadero significa que entre ellos no son aledaños, por lo que es seguro bondear
                        if sum(get_offset(available_bl[0], i[0])) > 1:
                            available_l.append(i[0])

                    # Si queda algún free l que respete la condición de ángulo
                    if len(available_l) > 0:
                        # Elegimos un número al azar
                        ix = np.random.randint(0, len(available_l))

                        # Agregamos el free l como bond a nuestro link inicial
                        self.current_bls[f"COORD_{coord}"].append(available_l[ix])

                        # Agregamos el free l al diccionario
                        self.current_bls[f"COORD_{available_l[ix]}"] = [coord]

                        # Actualizamos los valores de nuestro link inicial y el free l
                        self.matrix = change_val(self.matrix, coord, "bl")
                        self.matrix = change_val(self.matrix, available_l[ix], "bl")

                        # return matrix, current_bls

                    # Si no queda ningún free link que respete la condición, salimos
                    # else:
                    # return matrix, current_bls

                # Si no hay ningún free link, salimos
                # else:
                # return matrix, current_bls

            # Si no hay bl disponibles pero hay algún l
            elif len(surr_l) > 0:

                # Seleccionamos uno al azar
                ix = np.random.randint(0, len(surr_l))

                # Agregamos ambos al diccionario con su respectivo relativo
                self.current_bls[f"COORD_{coord}"] = [surr_l[ix][0]]
                self.current_bls[f"COORD_{surr_l[ix][0]}"] = [coord]

                # Actualizamos sus valores
                self.matrix = change_val(self.matrix, coord, "bl")
                self.matrix = change_val(self.matrix, surr_l[ix][0], "bl")

                # Removemos el free l con el que acabamos de bondear
                surr_l.pop(ix)

                # Si nuestra lista todavía tiene algún otro free l
                if len(surr_l) > 0:

                    available_l = []

                    # Para cada free l
                    for i in surr_l:

                        # Chequeamos si el offset entre el bl con el que enlazamos y el free l es mayor a 1
                        # Si esto es verdadero significa que entre ellos no son aledaños, por lo que es seguro bondear
                        if sum(get_offset(self.current_bls.get(f"COORD_{coord}")[0], i[0])) > 1:
                            available_l.append(i[0])

                    # Si queda algún free l que respete la condición de ángulo
                    if len(available_l) > 0:
                        # Seleccionamos uno al azar
                        ix = np.random.randint(0, len(available_l))

                        # Actualizamos el diccionario
                        self.current_bls[f"COORD_{coord}"].append(available_l[ix])
                        self.current_bls[f"COORD_{available_l[ix]}"] = [coord]

                        # Actualizamos su valores
                        self.matrix = change_val(self.matrix, available_l[ix], "bl")

                    # return matrix, current_bls

                # Si no, directamente retornamos la matriz
                # else:
                # return matrix, current_bls

            # else:
            # return matrix, current_bls

    def rebond(self):

        coords = get_all(self.matrix, "l")

        for c in coords:
            self.bonding(c)

    def run(self, iterations=None, analytics=False):

        if iterations is not None:

            res = {"num_h": [],
                   "num_l": [],
                   "num_bl": [],
                   "num_s": [],
                   "num_k": []}

            for i in tqdm(range(iterations), position=0):

                self.motion_H()
                self.motion_L()
                self.motion_K()
                self.composition()
                self.desintegrar()
                self.rebond()

                if analytics:
                    for i in ["h", "l", "bl", "s", "k"]:
                        res[f"num_{i}"].append(len(get_all(self.matrix, i)))

            return res

        else:

            self.motion_H()
            self.motion_L()
            self.motion_K()
            self.composition()
            self.desintegrar()



def ver_matrix_(matrix):

  coord_l = get_all(matrix, "l")
  coord_k = get_all(matrix, "k")
  coord_bl = get_all(matrix, "bl")
  coord_h = get_all(matrix, "h")
  coord_s = get_all(matrix, "s")

  plt.figure(figsize=(10, 10))

  plt.scatter(np.array(coord_l)[:,0], np.array(coord_l)[:,1], marker=(4,0,45), s=75, c="green", label="Link")
  plt.scatter(np.array(coord_k)[:,0], np.array(coord_k)[:,1], marker=(4,0,45), s=75, c="violet", label="Catalizador")
  if coord_bl != []:
    plt.scatter(np.array(coord_bl)[:,0], np.array(coord_bl)[:,1], marker=(4,0,45), s=75, c="orange", label="Bonded link")
  plt.scatter(np.array(coord_h)[:,0], np.array(coord_h)[:,1], marker=(4,0,45), s=75, c="white", label="Holes")
  plt.scatter(np.array(coord_s)[:,0], np.array(coord_s)[:,1], marker=(4,0,45), s=75, c="lightgreen", label="Sustrato")
  plt.axis("off")
  plt.legend(loc=(1, 0.7))
  plt.show()
