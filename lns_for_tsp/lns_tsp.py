import copy
import numpy as np
import random

# Function to calculate the travel distance between cities
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance += distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]
    return distance

# Function to calculate the total travel distance for the entire tour
def distance_point(tour, distance_matrix):
    # Ensure the tour is circular, so the last city connects back to the first one
    tour_shifted = np.roll(tour, shift=-1)  # Shift all elements in tour by -1
    # Ensure tour_shifted matches the bounds of the distance_matrix
    tour_shifted[-1] = tour[0]  # Connect the last city back to the first one
    return np.sum(distance_matrix[tour, tour_shifted])  # Sum the distances between consecutive cities
# Function to perform 2-opt local search to improve the tour
def local_search_2_opt(distance_matrix, route_distance, neighborhood_size, verbose=False):
    route, distance = route_distance
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                # Skip consecutive cities
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[i:j][::-1]  # Reverse the order of cities between i and j
                new_distance = distance_point(new_route, distance_matrix)
                if new_distance < distance:
                    route = new_route
                    distance = new_distance
                    improved = True
                    if verbose:
                        print(f"2-opt improved route with new distance: {round(distance, 2)}")
        if not improved:
            break

    return route, distance

# Function to remove cities randomly (used for Large Neighborhood Search)
def random_removal(city_tour, neighborhood_size):
    removed = random.sample(city_tour[1:], neighborhood_size)
    city_tour = [t for t in city_tour if t not in removed]
    return removed, city_tour

# Function for best insertion strategy (used for Large Neighborhood Search)
def best_insertion(removed_nodes, city_tour, distance_matrix):
    for node in removed_nodes:
        best_insertion_cost = float('inf')
        best_insertion_index = -1
        for i in range(1, len(city_tour) + 1):
            last_node = city_tour[i - 1]
            next_node = city_tour[i % len(city_tour)]
            insertion_cost = (distance_matrix[last_node, node] + distance_matrix[node, next_node] - distance_matrix[last_node, next_node])
            if insertion_cost < best_insertion_cost:
                best_insertion_cost = insertion_cost
                best_insertion_index = i
        city_tour.insert(best_insertion_index, node)
    return city_tour

# Function to perform Large Neighborhood Search for Truck and Drone
def large_neighborhood_search(distance_matrix, iterations=100, neighborhood_size=4, local_search=True, verbose=True):
    initial_tour = list(range(0, distance_matrix.shape[0]))
    random.shuffle(initial_tour)
    route = initial_tour.copy()
    distance = distance_point(route, distance_matrix)
    count = 0
    while count <= iterations:
        if verbose and count > 0:
            print(f'Iteration = {count}, Distance = {round(distance, 2)}')
        city_tour = route.copy()
        removed_nodes, city_tour = random_removal(city_tour, neighborhood_size)
        new_tour = best_insertion(removed_nodes, city_tour, distance_matrix)
        new_tour_distance = distance_point(new_tour, distance_matrix)
        if new_tour_distance < distance:
            route = new_tour
            distance = new_tour_distance
        count += 1
    route = route + [route[0]]  # Make the tour a round-trip
    route = [item + 1 for item in route]  # Adjusting for 1-based indexing
    if local_search:
        route, distance = local_search_2_opt(distance_matrix, [route, distance], -1, verbose)
    return route, distance

# Function to load city data and calculate distance matrix
def load_data():
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\lns\\data\\48.txt"
    data = np.loadtxt(file_path, dtype=int)
    N = len(data)
    dist_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i][j] = np.sqrt((data[i, 1] - data[j, 1]) ** 2 + (data[i, 2] - data[j, 2]) ** 2)
    return dist_matrix

# Function to dispatch drone based on probability and track movement
def dispatch_drone(route, distance_matrix, probability=0.1):
    drone_path = []
    vehicle_path = []
    total_distance = 0.0
    truck_position = 0  # Start from the first city
    drone_position = 0
    truck_speed = 1
    drone_speed = 1.5  # Drone speed is faster

    # Loop over the route to simulate truck and drone movements
    for i in range(len(route) - 1):
        truck_pos = route[truck_position]
        truck_next_pos = route[truck_position + 1]

        # Dispatch drone with probability
        if random.random() < probability:
            drone_path.append((truck_pos, truck_next_pos))
            vehicle_path.append(2)  # Drone travels here
            # Drone takes a shortcut to the next city
            drone_time = distance_matrix[truck_pos-1, truck_next_pos-1] / drone_speed  # Drone time
            total_distance += drone_time
        else:
            vehicle_path.append(1)  # Truck travels here
            truck_time = distance_matrix[truck_pos-1, truck_next_pos-1] / truck_speed  # Truck time
            total_distance += truck_time

        truck_position += 1  # Move to the next city for the truck

    return vehicle_path, drone_path, total_distance

# Main function to execute the problem and display results
def main():
    distance_matrix = load_data()
    route, distance = large_neighborhood_search(distance_matrix)

    # Dispatch drone and calculate total distance with vehicle and drone movement
    vehicle_path, drone_path, total_distance = dispatch_drone(route, distance_matrix)

    # Final result: Cities visited in order, vehicle path (1=Truck, 2=Drone), total distance
    print("Cities visited in order:", route)
    print("Vehicle path (1=Truck, 2=Drone):", vehicle_path)
    print("Total Distance:", total_distance)

# Execute the main function
if __name__ == "__main__":
    main()
