from functools import partial
from tkinter import *
from tkinter import messagebox
import time
import random
import sys

columnas = 0
filas = 0
tableroM = []
cont = 0


def sintetico(infoTablero):
    global filas
    global columnas
    global tableroM
    
    tableroM = infoTablero[0].strip().split(",")
    filas = int(tableroM[0])
    columnas = int(tableroM[1])
    center_coords = dict()

    best_individual =[(0,3),(0,0),(0,1),(0,2),(0,4),(2,1),(2,3),(2,4),(4,1),(3,0),(4,0),(4,2),(4,3),(4,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,2),(3,1),(3,2),(3,3),(3,4)]
    grid_size = filas
    list_size = filas * columnas
    
    for i in range(1, len(infoTablero)):
        linea= infoTablero[i].strip().split(",")
        
        center_coords[(int(linea[1])-1, int(linea[2])-1)] = int(linea[0])

        center_coords_keys = list(center_coords.keys())
        center_coords_vals = list(center_coords.values())
    max_swaps = max(center_coords_vals) - 1

    s = 0
    cum_sum = [0]
    for x in center_coords_vals:
        s = s + x
        cum_sum.append(s)

    # Todos los índices de dónde comienzan las islas.
    cum_sum_butlast = cum_sum[:-1]
    cum_sum_withmax = cum_sum.copy()
    cum_sum_withmax.append(list_size)

    max_islands = s
    max_waters = list_size - s

    # Una lista que contiene todas las coordenadas (x,y) correspondientes al tamaño de la cuadrícula.
    all_coords = [(x, y) for y in range(grid_size) for x in range(grid_size)]

    # Una lista que no contiene las coordenadas del centro.
    valid_coords = [x for x in all_coords if x not in center_coords]

    # Diccionario que contiene todos los bordes válidos para la coordenada correspondiente
    adjacencies = dict()
    for coord in all_coords:
        adjacents = []
        x, y = coord
        adjacents += [(x+1, y)] if x+1 < grid_size else []
        adjacents += [(x, y+1)] if y+1 < grid_size else []
        adjacents += [(x-1, y)] if x-1 >= 0 else []
        adjacents += [(x, y-1)] if y-1 >= 0 else []
        adjacencies[coord] = adjacents

    class NurikabeGA():

        def __init__(self, grid_size, center_coords, generations, print_interval):

            self.grid_size = grid_size
            self.center_coords = center_coords
            self.gene_pool = [(x, y) for y in range(self.grid_size)
                              for x in range(self.grid_size) if (x, y) not in self.center_coords]

            self.generations = generations

            self.print_interval = print_interval

        def geneticAlgorithm(self, pop_size, mating_pool_size, elite_size, mutation_rate, multi_objective_fitness=False):
            startTime = time.time()

            if multi_objective_fitness:
                populations = []
                for island in range(len(cum_sum_butlast)):
                    populations.append(Population(pop_size=pop_size, mating_pool_size=mating_pool_size, elite_size=elite_size, mutation_rate=mutation_rate, multi_objective_fitness=multi_objective_fitness, island_number=island))

                best_individual = Individual()
                best_fitness = 0
                best_generation = 0

                for i in range(0, self.generations):
                    for pop in populations:
                        for ind in pop.population:
                            fitness = ind.calculate_fitness()
                            if fitness > best_fitness:
                                best_individual.individual = ind.individual.copy()
                                best_fitness = fitness
                                best_generation = i
                        pop.breedPopulation()

                    if best_individual.isSolved():
                        print()
                        print("Nurikabe Solved in", time.time() - startTime, "seconds!")
                        print("Generation ", i, ": Best Fitness = ", best_fitness)
                        best_individual.printAsMatrix(best_individual.isSolved())
                        break

                    if i % self.print_interval == 0:
                        print()
                        print("Generation ", i, ": Best Fitness = ", best_fitness)
                        best_individual.printAsMatrix(best_individual.isSolved())

                    self.breedPopulations(populations)

            else:
                population = Population(
                    pop_size=pop_size, mating_pool_size=mating_pool_size, elite_size=elite_size, mutation_rate=mutation_rate)

                best_individual = Individual()
                best_fitness = 0
                best_generation = 0
                avg_fitness = 0

                for i in range(0, self.generations):
                    for ind in population.population:
                        fitness = ind.calculate_fitness()
                        avg_fitness += fitness
                        if fitness > best_fitness:
                            best_individual.individual = ind.individual.copy()
                            best_fitness = fitness
                            best_generation = i

                    if best_individual.isSolved():
                        best_individual.printAsMatrix(best_individual.isSolved())
                        break

                    if i % self.print_interval == 0:
                        best_individual.fixRange()
                        best_individual.printAsMatrix(best_individual.isSolved())

                    avg_fitness = 0

                    population.breedPopulation()

        def breedPopulations(self, populations):
            random.shuffle(populations)
            for x in range(len(populations)-1):
                cur_pop = populations[x]
                next_pop = populations[x+1]

                ind_num = 0
                for ind in populations[x].population:
                    breed_chance = random.random()
                    if breed_chance < 0.5:
                        ind = populations[x+1].population[ind_num]
                    else:
                        populations[x].population[ind_num] = ind
                    ind_num += 1



    class Population():

        def __init__(self, pop_size, mating_pool_size, elite_size, mutation_rate, multi_objective_fitness=False, island_number=-1):

            self.population = []

            self.pop_size = pop_size
            self.mating_pool_size = mating_pool_size
            self.elite_size = elite_size
            self.mutation_rate = mutation_rate
            self.multi_objective_fitness = multi_objective_fitness
            self.island_number = island_number

            for _ in range(pop_size):
                self.population.append(Individual(multi_objective_fitness))

        # - Selecting a random sample (like 100), then evaluating fitness and sorting them from best to worst.
        def getMatingPool(self):
            mating_pool = []

            sample = random.sample(self.population, k=self.mating_pool_size)

            if self.multi_objective_fitness:
                for individual in sample:
                    mating_pool.append([individual, individual.calculate_fitness()])
            else:
                for individual in sample:
                    mating_pool.append([individual, individual.calculate_fitness()])

            sorted_pool = sorted(mating_pool, key=lambda x: x[1], reverse=True)
            only_mates = [x for x, y in sorted_pool]

            return only_mates

        def breed(self, mating_pool):
            children = []
            elites = []

            for i in range(self.elite_size):
                children.append(mating_pool[i])
                elites.append(mating_pool[i])

            for i in range(self.mating_pool_size - self.elite_size):
                random_parent1 = random.randint(0, self.elite_size-1)
                random_parent2 = random.randint(0, self.mating_pool_size-1)

                random_chance = random.random()
                if random_chance < 0.5:
                    parent1 = elites[random_parent1]
                else:
                    parent1 = mating_pool[random.randint(0, self.mating_pool_size-1)]
                parent2 = mating_pool[random_parent2]

                matching_coords = [
                    x for x in parent1.individual if x in parent2.individual and x not in center_coords_keys]

                try:
                    random_coords = random.sample(matching_coords, k=2)

                    p2_a_index = parent2.index(random_coords[0])
                    p2_b_index = parent2.index(random_coords[1])

                    child = parent2.copy()
                    child[p2_a_index] = random_coords[1]
                    child[p2_b_index] = random_coords[0]

                    children.append(child)
                except:
                    pass

                children.append(Individual())

            for i in range(self.pop_size - self.mating_pool_size):
                children.append(Individual())

            return children

    #Mueve una isla seleccionada al azar del Padre 1 y la pone en el padre 2
    #También se agregó el intercambio de océanos.

        def single_island_crossover(self, mating_pool):
            children = []
            elites = []

            # Maintain some elites based on elite size
            for i in range(self.elite_size):
                children.append(mating_pool[i])
                elites.append(mating_pool[i])

            for i in range(self.mating_pool_size - self.elite_size):
                # Encuentra algunas padres al azar
                p1_random: Individual = random.choice(elites)
                p2_random: Individual = random.choice(mating_pool)

                #random_island_range elige un rango aleatorio como [0,5] o [5,10]
                p1_island_range = p1_random.random_island_range()

                # Encuentra los elementos reales en el rango y los convierte en un conjunto
                if len(p1_island_range) != 1:
                    p1_toSet = set(
                        p1_random.individual[p1_island_range[0]+1:p1_island_range[1]])
                    p2_toSet = set(
                        p2_random.individual[p1_island_range[0]+1:p1_island_range[1]])
                    avoid_range = range(p1_island_range[0] + 1, p1_island_range[1])
                else:
                    p1_toSet = set(
                        p1_random.individual[p1_island_range[0]:list_size])
                    p2_toSet = set(
                        p2_random.individual[p1_island_range[0]:list_size])
                    avoid_range = range(p1_island_range[0], list_size)

                p1_toList = list(p1_toSet)
                remainders = list(p2_toSet - p1_toSet)
                child = Individual()
                child.individual = p2_random.individual.copy()
                inner = 0

                for i in range(list_size):
                    if i in avoid_range:
                        child.individual[i] = p1_toList[inner]
                        inner += 1
                    else:
                        if child.individual[i] in p1_toList:
                            child.individual[i] = remainders.pop()

                children.append(child)

            for i in range(self.pop_size - self.mating_pool_size):
                children.append(Individual())

            return children

        # Todos los individuos
        # En este caso, deberían ser todos los hijos
        # Intercambia aleatoriamente a un hijo con un océano
        def mutate(self, children):
            for child in children:
                rand_chance = random.random()
                rand_chance_2 = random.random()
                squareOcean = child.findFirstOceanSquare()

                all_one_main_islands = [x for x in center_coords if center_coords[x] == 1]
                random_single_isolate = random.choice(all_one_main_islands) if all_one_main_islands else False

                if rand_chance < self.mutation_rate:
                    swaps_at_a_time = random.randint(1, max_swaps)
                    for _ in range(swaps_at_a_time):
                        random_land = random.randint(0, max_islands-1)
                        while random_land in cum_sum_butlast:
                            random_land = random.randint(0, max_islands-1)
                        random_ocean = random.randint(max_islands, list_size-1)
                        temp_coord = child.individual[random_land]
                        child.individual[random_land] = child.individual[random_ocean]
                        child.individual[random_ocean] = temp_coord
                else:
                    child.propogationMutation()
                    if rand_chance_2 < self.mutation_rate:
                        child.fixRange()
                    else:
                        child.mutateRange()

                if(squareOcean != 0):
                    child.fixASquare(squareOcean)

                if random_single_isolate:
                    child.isolateSingleIsland(random_single_isolate)

            return children

        def breedPopulation(self):
            mating_pool = self.getMatingPool()
            children = self.single_island_crossover(mating_pool)
            mutated = self.mutate(children)
            self.population = mutated
            return self


        def printAsMatrix(self, index):

            grid = [[0]*grid_size for i in range(grid_size)]

            islandNumber = 1

            for i in range(len(cum_sum)-1):
                for x, y in self.population[index].individual[cum_sum[i]:cum_sum[i+1]]:
                    # Assign the value in the grid
                    grid[x][y] = islandNumber
                islandNumber += 1

            for _ in grid:
                print(_)


    class Individual():


        def __init__(self, multi_objective_fitness=False):

            self.individual = []

            self.multi_objective_fitness = multi_objective_fitness

            isl = 0

            random_valid_coords = valid_coords.copy()
            random.shuffle(random_valid_coords)


            for i in range(list_size):

                if isl < len(cum_sum_butlast) and i == cum_sum_butlast[isl]:
                    self.individual.append(center_coords_keys[isl])
                    isl += 1
                else:
                    self.individual.append(random_valid_coords.pop())

            self.ocean_start_index = cum_sum[-1]
            self.empty_list = [[0 for x in range(grid_size)] for y in range(grid_size)]


            self.squares = []
            for i in range(grid_size-1):
                for j in range(grid_size-1):
                    self.squares.append(((i,j),(i+1,j),(i,j+1),(i+1,j+1)))


        def calculate_fitness(self, island_focus=-1):
            total_fitness = 0.0

            if self.allInRange():
                pass
            else:
                return 0

            oceans_fitness = self.connectedFitnessOcean()
            isolation_fitness = self.isIsolated()

            if oceans_fitness == max_waters:
                total_fitness += max_waters
            else:
                return oceans_fitness

            if self.isOceanSquare():
                return oceans_fitness - self.numOceanSquares()*4

            total_fitness += isolation_fitness
            if isolation_fitness == len(center_coords):
                pass
            else:
                return total_fitness

            if island_focus != -1:
                total_fitness += self.connectedFitness()
            else:
                total_fitness += self.connectedFitness()

            return total_fitness

        def calculate_overall_fitness(self):
            total_fitness = 0
            return total_fitness


        def isAdj(self, coord1, coord2):

            x1,y1 = coord1
            x2,y2 = coord2

            return (abs(x2-x1) + abs(y2-y1) == 1)

        def isAdjinList(self, coordlist, coord):
            for coordinate in coordlist:
                if self.isAdj(coordinate,coord):
                    return True
            return False

        # returns the coordinate from coordlist1 that is adj to coordlist2 otherwise returns 0
        def coordAdjbetweenTwoLists(self,coordlist1, coordlist2):
            for coord1 in coordlist1:
                for coord2 in coordlist2:
                    if(self.isAdj(coord1,coord2)):
                        return coord1
            return 0

        def prepareIslandLists(self):
            tempList = []
            combinedLists = []
            for i in range(len(cum_sum)-1):
                for coord in self.individual[cum_sum[i]:cum_sum[i+1]]:
                    tempList.append(coord)
                combinedLists.append(tempList)
                tempList = []
            return combinedLists


        def findConnected(self):
            islands = self.prepareIslandLists()
            connectedIslands = []
            coordsAdjinclCenter = []

            searching = True

            for island in islands:
                coordsAdjinclCenter.append(island.pop(0))
                while(searching):
                    adjCoord = self.coordAdjbetweenTwoLists(island,coordsAdjinclCenter)
                    if(adjCoord != 0):
                        coordsAdjinclCenter.append(island.pop(island.index(adjCoord)))
                    else:
                        searching = False

                connectedIslands.append(coordsAdjinclCenter)
                coordsAdjinclCenter = []
                searching = True
            return connectedIslands

        def findConnectedOcean(self):
            ocean = self.individual[self.ocean_start_index:len(self.individual)]
            connectedOceans = []
            coordsAdjinclCenter = []
            searching = True

            coordsAdjinclCenter.append(ocean.pop(0))
            while(searching):
                adjCoord = self.coordAdjbetweenTwoLists(ocean,coordsAdjinclCenter)
                if(adjCoord != 0):
                    coordsAdjinclCenter.append(ocean.pop(ocean.index(adjCoord)))
                else:
                    searching = False

            connectedOceans.append(coordsAdjinclCenter)
            return connectedOceans

        def findConnectedOceans2(self):
            searching = True
            coordsAdjtoFirst = []
            ocean = self.individual[cum_sum[-1]:]
            coordsAdjtoFirst.append(ocean.pop(0))
            while(searching):
                adjCoord = self.coordAdjbetweenTwoLists(ocean,coordsAdjtoFirst)
                if(adjCoord != 0):
                    coordsAdjtoFirst.append(ocean.pop(ocean.index(adjCoord)))
                else:
                    searching = False
            return coordsAdjtoFirst

        def connectedOceanFitness2(self):
            bestOceanSize = list_size - cum_sum[-1]
            connectedOcean = self.findConnectedOceans2()
            if(len(connectedOcean) == bestOceanSize):
                # double the points if its the right size
                return bestOceanSize * 2
            return len(connectedOcean)

        # Returns whether or not there is a square in the ocean
        def isOceanSquare(self):
            for square in self.squares:
                if(set(square).issubset(self.individual[cum_sum[-1]:])):
                    return True
            return False

        def findFirstOceanSquare(self):
            for square in self.squares:
                if(set(square).issubset(self.individual[cum_sum[-1]:])):
                    return square
            return 0

        def numOceanSquares(self):
            ct = 0
            for square in self.squares:
                if(set(square).issubset(self.individual[cum_sum[-1]:])):
                    ct += 1
            return ct

        def adjCoords(self, coord):
            adjCoords = []
            x,y = coord
            if (x > 0):
                adjCoords.append((x-1,y))
            if (y > 0):
                adjCoords.append((x,y-1))
            if (x < grid_size-1):
                adjCoords.append((x+1,y))
            if(y < grid_size-1):
                adjCoords.append((x,y+1))
            return adjCoords

        def isIsland(self,coord):
            if(self.individual.index(coord) < cum_sum[-1]):
                return True
            return False

        def isolateSingleIsland(self, coord):
            # Check all the adj if theyre islands
            adjIslands = []
            for coordinate in self.adjCoords(coord):
                if(self.isIsland(coordinate)):
                    adjIslands.append(coordinate)

            for island in adjIslands:
                randomOceanIndex = self.individual.index(random.choice(self.individual[cum_sum[-1]:]))
                tempIsland = island
                self.individual[self.individual.index(island)] = self.individual[randomOceanIndex]
                self.individual[randomOceanIndex] = tempIsland

        def inRange(self, centerValue, coord1, coord2):
            x1, y1 = coord1
            x2, y2 = coord2
            distance = abs(x2-x1) + abs(y2-y1)
            if (distance <= centerValue):
                return True
            return False


        def connectedFitness(self):
            connectedIslands = self.findConnected()
            bestIslandSizes = [x2 - x1 for (x1, x2) in zip(cum_sum[0:], cum_sum[1:])]
            bestScore = max(bestIslandSizes)
            connectedFitness = sum([-1 if len(cIsland) > sizes else len(cIsland) if len(cIsland) != sizes
            else bestScore for (cIsland, sizes) in zip(connectedIslands, bestIslandSizes)])

            return connectedFitness

        def connectedFitnessOcean(self):
            connectedOceans = self.findConnectedOcean()[0]
            connectedFitness = max_waters if len(connectedOceans) == max_waters else len(connectedOceans)

            return connectedFitness

        def connectedFitnessWeighted(self, island_number):
            connectedIsland = self.findConnected()[island_number]
            bestIslandSizes = [x2 - x1 for (x1, x2) in zip(cum_sum[0:], cum_sum[1:])]
            bestIslandSize = bestIslandSizes[island_number]
            islandFitness = bestIslandSize if len(connectedIsland) == bestIslandSize else len(connectedIsland) if len(connectedIsland) < bestIslandSize else len(connectedIsland) + (bestIslandSize - len(connectedIsland))
            return islandFitness - 1

        def isIsolated(self):
            # Las adyacencias de la isla solo deben contenerse a si misma oa un oceano.
            # Realiza un seguimiento de las iteraciones
            isl = 0

            # Incorporated a fitness value
            fitness_val = 0

            # Para cada coordenada en una isla, la posición adyacente de la coordenada debe estar dentro de sí misma o del océano.
            for island_start in cum_sum_butlast:
                island_end = cum_sum[isl+1]

                # island is a list or splice of coordinates corresponding to an island
                island = self.individual[island_start:island_end]
                other_islands = list(set(self.individual[0:cum_sum[-1]])-set(island))

                good_island = True

                all_adjacents = []
                for coord in island:
                    adjacents = adjacencies[coord]
                    for a in adjacents:
                        all_adjacents.append(a)
                all_adjacents_no_dupes = set(all_adjacents)
                for coord in island:
                    if coord not in all_adjacents_no_dupes and len(island) != 1:
                        good_island = False
                for a in all_adjacents_no_dupes:
                    if a in other_islands:
                        good_island = False

                if good_island:
                    fitness_val += 1

                isl += 1

            return fitness_val

        def islandsNotIsolated(self):
            # Las adyacencias de la isla solo deben contenerse a si misma oa un oceano.
            # Realiza un seguimiento de las iteraciones
            isl = 0
            isolated_islands = []
            for island_start in cum_sum_butlast:
                island_end = cum_sum[isl+1]

                island = self.individual[island_start:island_end]
                other_islands = list(set(self.individual[0:cum_sum[-1]])-set(island))

                good_island = True

                all_adjacents = []
                for coord in island:
                    adjacents = adjacencies[coord]
                    for a in adjacents:
                        all_adjacents.append(a)
                all_adjacents_no_dupes = set(all_adjacents)
                for coord in island:
                    if coord not in all_adjacents_no_dupes and len(island) != 1:
                        good_island = False
                for a in all_adjacents_no_dupes:
                    if a in other_islands:
                        good_island = False

                if good_island:
                    pass
                else:
                    isolated_islands.append(island)

                isl += 1

            return isolated_islands

        def random_island_range(self):
            island_start_index = random.choice(range(len(cum_sum_withmax)))

            return cum_sum_withmax[island_start_index:island_start_index + 2]

        def shortIsland(self):
            islands = self.findConnected()

            island_number = 0
            for island in islands:
                if len(island) != center_coords_vals[island_number]:
                    return island_number,island
                else:
                    island_number += 1
            return -1,[]


        def addToShortIsland(self):

            short_island_index,connected_islands = self.shortIsland()

            if short_island_index == -1:
                return ()

            short_island = self.individual[cum_sum[short_island_index]:cum_sum[short_island_index+1]]

            replacement_coordinates = set(short_island) - set(connected_islands)

            coords_all_adjacent = set([x for sub in [adjacencies[coord] for coord in connected_islands] for x in sub])
            valid_adjacents_in_ocean = coords_all_adjacent.intersection(set(self.individual[self.ocean_start_index:len(self.individual)]))

            return (list(replacement_coordinates), list(valid_adjacents_in_ocean))

        def propogationMutation(self):
            if len(self.addToShortIsland()) != 0:
                replacement_coordinates, valid_adjacents_in_ocean = self.addToShortIsland()
                random.shuffle(valid_adjacents_in_ocean)

                for coord in replacement_coordinates:
                    my_index = self.individual.index(coord)
                    if valid_adjacents_in_ocean:
                        ocean_index = self.individual.index(valid_adjacents_in_ocean.pop())
                        temp_coord = coord
                        self.individual[my_index] = self.individual[ocean_index]
                        self.individual[ocean_index] = temp_coord
            else:
                pass
            
        def printAsMatrix(self, resuelto):
            global cont 
            
            cont = cont + 1
            island_number = 1
            ct = 0
            for x,y in self.individual:
                if ct in cum_sum_butlast[1:]:
                    island_number += 1
                elif ct >= cum_sum[-1]:
                    island_number = 0

                self.empty_list[x][y] = island_number
                ct += 1
            
            tablero = Tk()
            tablero.title("Nurikabe Sintetico")
            texto = ("Numero de pasos para llegar a la solucion: " + str(cont))
            titulo = Label(tablero, text=texto, bg="Black", fg="white", font=("Verdana",12))
            titulo.pack(fill=X)
            matriz = Frame(tablero)
            matriz.pack()
            
            fil=0

            for row in self.empty_list:
                col=0
                for k in row:
                    if k != 0:
                        n_button = Button(matriz, width=7, height=3, background="white" ,foreground="black")
                        n_button.grid(row=fil, column=col)
                    else:
                        n_button = Button(matriz, width=7, height=3, background="black" ,foreground="red")
                        n_button.grid(row=fil, column=col)
                    col = col+1
                fil = fil+1
            Label(tablero).pack()
            
            if resuelto:
                text1 = Label(tablero, text="Solucion del juego", font=("Verdana",18))
                text1.pack(side=BOTTOM)
    
        def isSolved(self):
            bestIslandSizes = [x2 - x1 for (x1, x2) in zip(cum_sum[0:], cum_sum[1:])]
            bestScore = max(bestIslandSizes)
            largest_fitness_possible = len(center_coords) + max_waters + (bestScore * len(center_coords))
            if self.isIsolated() == len(center_coords) and self.connectedFitnessOcean() == max_waters and not self.isOceanSquare() and self.calculate_fitness() == largest_fitness_possible:
                return True

        def fixASquare(self, square):

            if (square != 0):

                random_ocean_coord = random.choice(square)

                random_ocean_index = self.individual.index(random_ocean_coord)

                closest_coord_range = self.closestMainIsland(random_ocean_coord)

                if closest_coord_range:

                    random_land_index = random.choice(closest_coord_range)
                    random_land_coord = self.individual[random_land_index]
                    self.individual[random_land_index] = random_ocean_coord
                    self.individual[random_ocean_index] = random_land_coord

        def closestMainIsland(self, coord):
            coordX, coordY = coord
            closest_distance = sys.maxsize
            closest_coord_range = []
            next_index = 1
            for main_coord_index in cum_sum_butlast:
                mainX, mainY = self.individual[main_coord_index]
                # distance = math.sqrt(abs(coordX - mainX)**2 + abs(coordY - mainY)**2) #Euclidian
                distance = abs(coordX - mainX) + abs(coordY - mainY) #manhattan
                if distance < closest_distance:
                    closest_distance = distance
                    closest_coord_range = range(main_coord_index+1, cum_sum[next_index])
                next_index += 1
            return closest_coord_range

        def fixRange(self):
            for i in range(len(cum_sum_butlast)):
                center = self.individual[cum_sum[i]]
                max_size = center_coords[center]
                for coord in self.individual[cum_sum[i]+1:cum_sum[i+1]]:
                    if max_size != 1:
                        if self.inRange(max_size, center, coord):
                            pass
                        else:
                            in_range_oceans = [x for x in self.individual[cum_sum[-1]:] if self.inRange(max_size, center, x)]
                            if in_range_oceans:
                                random_ocean = random.choice(in_range_oceans)
                                rand_ocean_index = self.individual.index(random_ocean)
                                coord_index = self.individual.index(coord)

                                self.individual[rand_ocean_index] = coord
                                self.individual[coord_index] = random_ocean

        def allInRange(self):
            for i in range(len(cum_sum_butlast)):
                center = self.individual[cum_sum[i]]
                max_size = center_coords[center]
                for coord in self.individual[cum_sum[i]+1:cum_sum[i+1]]:
                    if max_size != 1:
                        if self.inRange(max_size, center, coord):
                            pass
                        else:
                            return False
            return True

        def mutateRange(self):
            for i in range(len(cum_sum_butlast)):
                center = self.individual[cum_sum[i]]
                max_size = center_coords[center]
                coords = self.individual[cum_sum[i]+1:cum_sum[i+1]]
                if coords:
                    coord = random.choice(coords)
                    if max_size != 1:
                        in_range_oceans = [x for x in self.individual[cum_sum[-1]:] if self.inRange(max_size, center, x)]
                        if in_range_oceans:
                            random_ocean = random.choice(in_range_oceans)
                            rand_ocean_index = self.individual.index(random_ocean)
                            coord_index = self.individual.index(coord)

                            self.individual[rand_ocean_index] = coord
                            self.individual[coord_index] = random_ocean


    def main():
        nurikabe = NurikabeGA(grid_size=grid_size, center_coords=center_coords, generations=100000, print_interval=5)
        nurikabe.geneticAlgorithm(
            pop_size=2000, mating_pool_size=1000, elite_size=100, mutation_rate=0.5, multi_objective_fitness=False)

        return 0

    main()
