import numpy as np
import random

# Function: Tour Distance
def distance_calc(distance_matrix, tour):
    distance = 0
    for i in range(len(tour) - 1):
        distance += distance_matrix[tour[i], tour[i + 1]]
    return distance

# Function: Add Drone to Route
def add_drone_to_route(route, distance_matrix, drone_probability=0.3):
    vehicle_assignment = []
    truck_path = []
    drone_path = []
    i = 0

    while i < len(route) - 1:
        truck_path.append(route[i])
        vehicle_assignment.append(1)  # Truck visits this city

        # Xác suất phóng drone tại một thành phố
        if random.random() < drone_probability and i < len(route) - 2:
            drone_path.append(route[i + 1])  # Drone thăm thành phố tiếp theo
            vehicle_assignment.append(2)  # Drone thực hiện bước này
            i += 1  # Bỏ qua thành phố này trong tuyến của truck

        i += 1

    truck_path.append(route[-1])  # Quay về thành phố đầu tiên
    vehicle_assignment.append(1)

    # Tính toán quãng đường
    truck_distance = distance_calc(distance_matrix, truck_path)
    drone_distance = distance_calc(distance_matrix, drone_path)
    total_distance = truck_distance + drone_distance

    return truck_path, drone_path, vehicle_assignment, total_distance

# Function: Large Neighborhood Search (Giữ nguyên)
def large_neighborhood_search(distance_matrix, iterations=100, neighborhood_size=4, verbose=True):
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

    return route + [route[0]], best_distance

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

    # Tối ưu hóa tuyến đường của truck
    truck_route, truck_distance = large_neighborhood_search(distance_matrix, iterations=100, neighborhood_size=4, verbose=True)

    # Thêm drone vào tuyến đường
    truck_path, drone_path, vehicle_assignment, total_distance = add_drone_to_route(truck_route, distance_matrix, drone_probability=0.3)

    # In kết quả
    print("\nOptimal route (truck):", truck_path)
    print("Drone visits:", drone_path)
    print("Vehicle assignment (1=truck, 2=drone):", vehicle_assignment)
    print("Total distance (truck + drone):", total_distance)

if __name__ == "__main__":
    main()
