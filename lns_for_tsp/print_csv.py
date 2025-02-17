import numpy as np
import random
import csv
import time

# Function: Tour Distance
def distance_calc(distance_matrix, tour):
    distance = 0
    for i in range(len(tour) - 1):
        distance += distance_matrix[tour[i], tour[i + 1]]
    return distance

# Function: Calculate Total Distance for Truck Only
def calculate_truck_only_distance(route, distance_matrix):
    truck_distance = distance_calc(distance_matrix, route + [route[0]])
    return truck_distance

# Function: 2-opt Local Search
def local_search_2_opt(route, distance_matrix):
    best_distance = distance_calc(distance_matrix, route + [route[0]])
    best_route = route[:]
    for i in range(len(route) - 1):
        for j in range(i + 2, len(route)):
            if i == 0 and j == len(route) - 1:
                continue  # Skip reversing the entire tour
            new_route = route[:i] + route[i:j][::-1] + route[j:]
            new_distance = distance_calc(distance_matrix, new_route + [new_route[0]])
            if new_distance < best_distance:
                best_distance = new_distance
                best_route = new_route[:]
    return best_route, best_distance

# Function: 3-opt Local Search
def local_search_3_opt(route, distance_matrix):
    best_distance = distance_calc(distance_matrix, route + [route[0]])
    best_route = route[:]
    for i in range(len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            for k in range(j + 1, len(route)):
                # Generate all possible 3-opt moves
                new_routes = [
                    route[:i] + route[i:j] + route[j:k][::-1] + route[k:],
                    route[:i] + route[j:k] + route[i:j] + route[k:],
                    route[:i] + route[j:k][::-1] + route[i:j][::-1] + route[k:],
                ]
                for new_route in new_routes:
                    new_distance = distance_calc(distance_matrix, new_route + [new_route[0]])
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_route = new_route[:]
    return best_route, best_distance

def add_drone_to_route(route, distance_matrix, drone_probability=0.4):
    vehicle_assignment = []
    truck_path = []
    drone_path = []
    drone_return_distance = 0  # Quãng đường drone quay về
    i = 0
    while i < len(route) - 1:
        truck_path.append(route[i])
        vehicle_assignment.append(1)  # Truck visits this city
        # Xác suất phóng drone tại một thành phố
        if random.random() < drone_probability and i < len(route) - 2:
            drone_start = route[i]         # Nơi drone được phóng
            drone_target = route[i + 1]      # Thành phố drone thăm
            drone_return = route[i + 2]      # Thành phố drone quay về (truck sẽ tới)
            # Thêm thành phố drone thăm vào drone_path
            drone_path.append(drone_target)
            vehicle_assignment.append(2)     # Drone thực hiện bước này
            # Tính quãng đường drone quay về
            drone_return_distance += distance_matrix[drone_target][drone_return]
            i += 1  # Bỏ qua thành phố mà drone đã thăm trong tuyến của truck
        i += 1
    truck_path.append(route[-1])  # Quay về thành phố đầu tiên
    vehicle_assignment.append(1)
    # Tính toán quãng đường
    truck_distance = distance_calc(distance_matrix, truck_path)
    drone_distance = distance_calc(distance_matrix, drone_path) + drone_return_distance
    total_distance = truck_distance + drone_distance
    return truck_path, drone_path, vehicle_assignment, total_distance

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

def large_neighborhood_search(distance_matrix, iterations=200, neighborhood_size=4, verbose=True):
    """
    Thực hiện thuật toán Large Neighborhood Search (LNS) với 2 giai đoạn:
      - Giai đoạn xóa & chèn (removal & insertion)
      - Giai đoạn local search (2-opt/3-opt)
    
    Trong giai đoạn local search, mỗi vòng lặp (interaction) sẽ được ghi log vào file CSV.
    """
    num_cities = distance_matrix.shape[0]
    route = list(range(num_cities))
    random.shuffle(route)

    # Bắt đầu từ city 0
    zero_index = route.index(0)
    route = route[zero_index:] + route[:zero_index]
    best_distance = distance_calc(distance_matrix, route + [route[0]])

    if verbose:
        print("Initial route:", route)
        print("Initial distance:", best_distance)

    # Giai đoạn 1: Xóa & Chèn
    for iteration in range(iterations):
        removed_nodes, partial_tour = random_removal(route, neighborhood_size)
        new_tour = best_insertion(removed_nodes, partial_tour, distance_matrix)
        new_distance = distance_calc(distance_matrix, new_tour + [new_tour[0]])
        if new_distance < best_distance:
            route = new_tour
            best_distance = new_distance

    if verbose:
        print("\nAfter initial removal & insertion phase:")
        print("Route:", route)
        print("Distance:", best_distance)

    # Chuẩn bị ghi file CSV cho giai đoạn 2 (Local Search)
    csv_filename = "results.csv"
    fieldnames = ["Algorithm", "FileName", "Sum Interaction", "Interaction", "Executing time", "Fitness"]
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Giai đoạn 2: Dùng 2-opt & 3-opt
        for iteration in range(iterations):
            start_time = time.time()
            # Ở đây, với điều kiện if random.random() < 1, luôn luôn chọn 2-opt.
            # Bạn có thể thay đổi điều kiện để xen kẽ sử dụng 3-opt.
            if random.random() < 1:
                new_tour, new_distance = local_search_2_opt(route, distance_matrix)
                if verbose:
                    print(f"Iteration {iteration}: Using 2-opt, Distance = {new_distance}")
            else:
                new_tour, new_distance = local_search_3_opt(route, distance_matrix)
                if verbose:
                    print(f"Iteration {iteration}: Using 3-opt, Distance = {new_distance}")
            exec_time = time.time() - start_time

            if new_distance < best_distance:
                route = new_tour
                best_distance = new_distance

            # Ghi log thông tin vòng lặp vào file CSV
            writer.writerow({
                "Algorithm": "LNS",
                "FileName": r"D:\Tran Hoang Vu\Lab\VRDP\Large Nearest Search (LNS)\lns\TSP_with_drone.py",
                "Sum Interaction": iterations,
                "Interaction": iteration,
                "Executing time": exec_time,
                "Fitness": best_distance
            })

    # Đảm bảo route quay về thành phố 0
    zero_index = route.index(0)
    route = route[zero_index:] + route[:zero_index]

    return route + [route[0]], best_distance

def main():
    # Thay đổi file_path theo dữ liệu của bạn
    # file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\lns\\data\\48.txt"
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\lns\\data\\15.txt"
    
    distance_matrix = load_data(file_path)
    # Tối ưu hóa tuyến đường của truck
    truck_route, truck_distance = large_neighborhood_search(distance_matrix, iterations=200, neighborhood_size=4, verbose=True)

    # Thêm drone vào tuyến đường
    truck_path, drone_path, vehicle_assignment, total_distance = add_drone_to_route(truck_route, distance_matrix, drone_probability=0.4)

    # Tính tổng quãng đường của truck
    truck_only_distance = calculate_truck_only_distance(truck_path, distance_matrix)

    # In kết quả ra màn hình
    print("\nOptimal route (starting from 0):", truck_path)
    print("Total distance (truck):", truck_only_distance)
    print("Vehicle assignment (1=truck, 2=drone):", vehicle_assignment)

if __name__ == "__main__":
    main()
