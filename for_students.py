from itertools import compress
import random
import time
import matplotlib.pyplot as plt
from data import *


class ManageGA:
    def __init__(self, items, backpack_capacity):
        self.__items = items
        self.__backpack_capacity = backpack_capacity
        self.__population_size = 100
        self.__individual_size = len(items)
        self.__generations = 200
        self.__n_selection = 70
        self.__n_elite = 1
        self.__current_population = [[random.choice([True, False]) for _ in range(self.__individual_size)] for _ in range(self.__population_size)]
        self.__history_populations = [self.__current_population]
        self.__best_history_fitness = []

    def __fitness_individual(self, individual):
        total_weight = sum(compress(self.__items['Weight'], individual))
        if total_weight > self.__backpack_capacity:
            return 0
        return sum(compress(items['Value'], individual))

    def __fitness_all(self):
        fitness_values = []
        fitness_sum = 0
        for individual in self.__current_population:
            fitness_individual = self.__fitness_individual(individual)
            fitness_values.append(fitness_individual)
            fitness_sum += fitness_individual

        return fitness_sum, fitness_values

    def __roulette_selection(self, number):     #zrobic selekcje turniejową
        fitness_sum, fitness_values = self.__fitness_all()
        for i in range(0, self.__population_size):
            fitness_values[i] /= fitness_sum

        selected = random.choices(self.__current_population, weights=fitness_values, k=number)
        return selected

    def __tournament_selection(self, number):
        selected = []
        for _ in range(number):
            players = []
            for _ in range(2):
                index = random.randint(0, self.__population_size-1)
                players.append(self.__current_population[index])

            if self.__fitness_individual(players[0]) > self.__fitness_individual(players[1]):
                selected.append(players[0])
            else:
                selected.append(players[1])

        return selected

    def __create_new_generation(self, parents): # zrobić losowość
        parents1 = [parents[i][:self.__individual_size//2 + 1] for i in range(self.__n_selection)]
        parents2 = [parents[i][self.__individual_size//2 + 1:self.__individual_size] for i in range(self.__n_selection)]

        children = []
        for j in range(self.__n_selection):
            child = parents1[j] + parents2[self.__n_selection-j-1]
            children.append(child)

        return children

    def __create_new_generation_advanced(self, parents):
        children = []
        for i in range(self.__n_selection//2):

            parents_parts_1, parents_parts_2 = [], []
            index = random.randint(1, self.__individual_size - 1)
            for j in range(2):
                parents_parts_1.append(parents[i*2+j][:index])
                parents_parts_2.append(parents[i*2+j][index:])

            children.append(parents_parts_1[0]+parents_parts_2[1])
            children.append(parents_parts_1[1] + parents_parts_2[0])

        return children

    def __mutation(self, children): # dla każdej pozycji BER
        for child in children:
            index = random.randint(0, self.__individual_size-1)
            if child[index]:
                child[index] = False
            else:
                child[index] = True

    def __mutation_advanced(self, children):
        barrier = 6
        for child in children:
            for i in range(0, self.__individual_size - 1):
                ber = random.randint(1, 100)
                if ber < barrier:
                    child[i] = not child[i]

    def __change_generations(self, children):
        rest = self.__tournament_selection(self.__population_size-self.__n_selection - self.__n_elite)
        elite, elite_fitness = self.__find_best_individual()
        population = children + rest + [elite]
        self.__current_population = population

    def __find_best_individual(self):
        best_individual = None
        best_individual_fitness = -1
        for individual in self.__current_population:
            individual_fitness = self.__fitness_individual(individual)
            if individual_fitness > best_individual_fitness:
                best_individual = individual
                best_individual_fitness = individual_fitness

        return best_individual, best_individual_fitness

    def get_history_population(self):
        return self.__history_populations

    def get_best_history_fitness(self):
        return self.__best_history_fitness

    def perform_GA(self):

        best_individual, best_individual_fitness = None, None

        for r in range(self.__generations):
            parents = self.__tournament_selection(self.__n_selection)
            #parents = self.__tournament_selection(self.__n_selection)
            children = self.__create_new_generation_advanced(parents)
            #children = self.__create_new_generation(parents)
            #self.__mutation(children)
            self.__mutation_advanced(children)
            self.__change_generations(children)
            self.__history_populations.append(self.__current_population)
            best_individual, best_individual_fitness = self.__find_best_individual()

            if r == 0:
                self.__best_history_fitness.append(best_individual_fitness)
            elif best_individual_fitness > self.__best_history_fitness[r-1]:
                self.__best_history_fitness.append(best_individual_fitness)
            else:
                self.__best_history_fitness.append(self.__best_history_fitness[r-1])

        print('Best solution:', list(compress(self.__items['Name'], best_individual)))
        print('Best solution value:', best_individual_fitness)


items, knapsack_max_capacity = get_big()

start_time = time.time()
manager = ManageGA(items, knapsack_max_capacity)
manager.perform_GA()

end_time = time.time()
total_time = end_time - start_time

print('Time: ', total_time)


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


x = []
y = []
top_best = 10
for i, population in enumerate(manager.get_history_population()):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(manager.get_best_history_fitness(), 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
