import numpy as np
import random
import copy

# Function: Tour Distance
def distance_calc(distance_matrix, tour):
    distance = 0
    for i in range(len(tour) - 1):
        distance += distance_matrix[tour[i], tour[i + 1]]
    return distance

# Function: 2-opt (Local Search)
def local_search_2_opt(distance_matrix, city_tour, iterations=100, verbose=True):
    if verbose:
        print("\nLocal Search (2-opt)")
    best_tour = city_tour[:]
    best_distance = distance_calc(distance_matrix, best_tour)
    for _ in range(iterations):
        i, j = sorted(random.sample(range(1, len(best_tour) - 1), 2))
        new_tour = best_tour[:i] + list(reversed(best_tour[i:j + 1])) + best_tour[j + 1:]
        new_distance = distance_calc(distance_matrix, new_tour)
        if new_distance < best_distance:
            best_tour = new_tour
            best_distance = new_distance
    return best_tour, best_distance

# Function: 3-opt (Local Search)
def local_search_3_opt(distance_matrix, city_tour, iterations=100, verbose=True):
    if verbose:
        print("\nLocal Search (3-opt)")
    best_tour = city_tour[:]
    best_distance = distance_calc(distance_matrix, best_tour)
    for _ in range(iterations):
        i, j, k = sorted(random.sample(range(1, len(best_tour) - 1), 3))
        new_tour = (
            best_tour[:i]
            + list(reversed(best_tour[i:j + 1]))
            + list(reversed(best_tour[j + 1:k + 1]))
            + best_tour[k + 1:]
        )
        new_distance = distance_calc(distance_matrix, new_tour)
        if new_distance < best_distance:
            best_tour = new_tour
            best_distance = new_distance
    return best_tour, best_distance

# Function: Adaptive Local Search (Xen kẽ 2-opt và 3-opt)
def adaptive_local_search(distance_matrix, city_tour, iterations=100, verbose=True, probability_2opt=0.3):
    if verbose:
        print("\nAdaptive Local Search")
    
    best_tour = city_tour[:]
    best_distance = distance_calc(distance_matrix, best_tour)

    for _ in range(iterations):
        # Dùng xác suất để quyết định có sử dụng 2-opt hay 3-opt
        if random.random() < probability_2opt:
            new_tour, new_distance = local_search_2_opt(distance_matrix, best_tour, iterations=1, verbose=False)
        else:
            new_tour, new_distance = local_search_3_opt(distance_matrix, best_tour, iterations=1, verbose=False)
        
        if new_distance < best_distance:
            best_tour = new_tour
            best_distance = new_distance
    
    return best_tour, best_distance

# Function: Random Removal
def random_removal(city_tour, neighborhood_size):
    removed = random.sample(city_tour, neighborhood_size)
    remaining_tour = [city for city in city_tour if city not in removed]
    return removed, remaining_tour

# Function: Best Insertion
def best_insertion(removed_nodes, city_tour, distance_matrix):
    for node in removed_nodes:
        best_cost = float("inf")
        best_index = -1
        for i in range(len(city_tour)):
            prev_node = city_tour[i - 1]
            next_node = city_tour[i]
            insertion_cost = (
                distance_matrix[prev_node, node]
                + distance_matrix[node, next_node]
                - distance_matrix[prev_node, next_node]
            )
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_index = i
        city_tour.insert(best_index, node)
    return city_tour

# Function: Large Neighborhood Search with Adaptive Local Search
def large_neighborhood_search(distance_matrix, iterations=100, neighborhood_size=4, local_search=True, verbose=True):
    num_cities = distance_matrix.shape[0]
    route = list(range(num_cities))
    random.shuffle(route)
    best_distance = distance_calc(distance_matrix, route + [route[0]])

    if verbose:
        print("Initial route:", route)
        print("Initial distance:", best_distance)

    for iteration in range(iterations):
        removed_nodes, partial_tour = random_removal(route, neighborhood_size)
        new_tour = best_insertion(removed_nodes, partial_tour, distance_matrix)
        new_distance = distance_calc(distance_matrix, new_tour + [new_tour[0]])

        if verbose:
            print(f"Iteration {iteration}: Distance = {new_distance}")

        if new_distance < best_distance:
            route = new_tour
            best_distance = new_distance

    if local_search:
        route, best_distance = adaptive_local_search(distance_matrix, route, iterations=10, verbose=verbose)

    return route + [route[0]], best_distance

# Function: Load Data
def load_data(file_path):
    data = np.loadtxt(file_path, dtype=float)
    num_cities = len(data)
    distance_matrix = np.zeros((num_cities, num_cities), dtype=float)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.sqrt((data[i, 1] - data[j, 1]) ** 2 + (data[i, 2] - data[j, 2]) ** 2)
    return distance_matrix

# Main Function
def main():
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\lns\\data\\48.txt"
    distance_matrix = load_data(file_path)
    route, distance = large_neighborhood_search(distance_matrix, iterations=100, neighborhood_size=4, verbose=True)
    print("\nOptimal route for truck:", route)
    print("Optimal distance for truck:", distance)

if __name__ == "__main__":
    main()
