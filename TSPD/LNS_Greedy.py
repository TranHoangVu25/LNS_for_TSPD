import numpy as np
import random

# Function: Tính khoảng cách của tour
def distance_calc(distance_matrix, tour):
    distance = 0
    for i in range(len(tour) - 1):
        distance += distance_matrix[tour[i], tour[i + 1]]
    return distance

# Function: Tính tổng khoảng cách của truck (với quay lại điểm đầu)
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
                continue  # Bỏ qua đảo ngược tour toàn phần

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
                # Sinh ra các cách chuyển 3-opt khả dĩ
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

# Function: Thêm drone vào tour (chỉ là ví dụ minh họa)
def add_drone_to_route(route, distance_matrix, drone_probability=0.4):
    vehicle_assignment = []
    truck_path = []
    drone_path = []
    drone_return_distance = 0  # Quãng đường drone quay về
    i = 0

    while i < len(route) - 1:
        truck_path.append(route[i])
        vehicle_assignment.append(1)  # Truck thăm thành phố này

        # Với xác suất nhất định, phóng drone từ thành phố hiện tại
        if random.random() < drone_probability and i < len(route) - 2:
            # Drone bay từ thành phố i sang i+1 rồi quay về thành phố i+2
            drone_target = route[i + 1]
            drone_path.append(drone_target)
            vehicle_assignment.append(2)  # Drone thực hiện bước này

            # Tính quãng đường drone quay về
            drone_return_distance += distance_matrix[drone_target][route[i + 2]]
            i += 1  # Bỏ qua thành phố mà drone đã thăm

        i += 1

    truck_path.append(route[-1])  # Quay về điểm cuối
    vehicle_assignment.append(1)

    # Tính khoảng cách
    truck_distance = distance_calc(distance_matrix, truck_path)
    drone_distance = distance_calc(distance_matrix, drone_path) + drone_return_distance
    total_distance = truck_distance + drone_distance

    return truck_path, drone_path, vehicle_assignment, total_distance

# --- THAY ĐỔI PHẦN "DESTROY" --- #
# Function: Greedy Removal thay vì Random Removal
def greedy_removal(city_tour, neighborhood_size, distance_matrix):
    """
    Tính “saving” khi loại bỏ từng node (ngoại trừ depot - giả sử là node 0 ở vị trí đầu)
    và loại bỏ các node có saving cao nhất.
    """
    removal_candidates = []
    # Bắt đầu từ index 1 để không loại bỏ depot
    for i in range(1, len(city_tour)):
        prev_index = i - 1
        next_index = (i + 1) % len(city_tour)  # tour là vòng (sẽ kết thúc tại depot)
        saving = (distance_matrix[city_tour[prev_index], city_tour[i]] +
                  distance_matrix[city_tour[i], city_tour[next_index]] -
                  distance_matrix[city_tour[prev_index], city_tour[next_index]])
        removal_candidates.append((city_tour[i], saving))
    
    # Sắp xếp theo saving giảm dần (saving càng cao => loại bỏ càng có lợi)
    removal_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Chọn neighborhood_size node có saving cao nhất
    removed_nodes = [node for node, _ in removal_candidates[:neighborhood_size]]
    remaining_tour = [city for city in city_tour if city not in removed_nodes]
    return removed_nodes, remaining_tour

# Function: Best Insertion (chèn lại theo cách tham lam)
def best_insertion(removed_nodes, city_tour, distance_matrix):
    for node in removed_nodes:
        best_cost = float("inf")
        best_index = -1
        for i in range(len(city_tour)):
            prev_node = city_tour[i - 1]
            next_node = city_tour[i]
            insertion_cost = (distance_matrix[prev_node, node] +
                              distance_matrix[node, next_node] -
                              distance_matrix[prev_node, next_node])
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_index = i
        city_tour.insert(best_index, node)
    return city_tour

# Function: Load Data từ file
def load_data(file_path):
    data = np.loadtxt(file_path, dtype=float)
    num_cities = len(data)
    distance_matrix = np.zeros((num_cities, num_cities), dtype=float)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.sqrt((data[i, 1] - data[j, 1]) ** 2 + (data[i, 2] - data[j, 2]) ** 2)
    return distance_matrix

# --- GREEDY LARGE NEIGHBORHOOD SEARCH --- #
def greedy_large_neighborhood_search(distance_matrix, iterations=200, neighborhood_size=4, verbose=True):
    num_cities = distance_matrix.shape[0]
    route = list(range(num_cities))
    random.shuffle(route)

    # Đảm bảo bắt đầu từ thành phố 0 (depot)
    zero_index = route.index(0)
    route = route[zero_index:] + route[:zero_index]

    best_distance = distance_calc(distance_matrix, route + [route[0]])

    if verbose:
        print("Initial route:", route)
        print("Initial distance:", best_distance)

    # Vòng lặp destroy & repair sử dụng greedy removal và best insertion
    for iteration in range(iterations):
        removed_nodes, partial_tour = greedy_removal(route, neighborhood_size, distance_matrix)
        new_tour = best_insertion(removed_nodes, partial_tour, distance_matrix)
        new_distance = distance_calc(distance_matrix, new_tour + [new_tour[0]])

        if new_distance < best_distance:
            route = new_tour
            best_distance = new_distance

    if verbose:
        print("\nAfter initial greedy removal/insertion iterations:")
        print("Route:", route)
        print("Distance:", best_distance)

    # Tiếp tục cải thiện tour bằng 2-opt & 3-opt (local search)
    for iteration in range(iterations):  
        if random.random() < 1:  # Luôn chọn 2-opt trong ví dụ này
            new_tour, new_distance = local_search_2_opt(route, distance_matrix)
            if verbose:
                print(f"Iteration {iteration}: Using 2-opt, Distance = {new_distance}")
        else:  
            new_tour, new_distance = local_search_3_opt(route, distance_matrix)
            if verbose:
                print(f"Iteration {iteration}: Using 3-opt, Distance = {new_distance}")

        if new_distance < best_distance:
            route = new_tour
            best_distance = new_distance

    # Quay lại sao cho tour bắt đầu từ depot (0)
    zero_index = route.index(0)
    route = route[zero_index:] + route[:zero_index]

    return route + [route[0]], best_distance

def main():
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\lns\\data\\48.txt"
    # file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\lns\\data\\15.txt"
    
    distance_matrix = load_data(file_path)
    # Tối ưu hóa tuyến đường của truck sử dụng Greedy LNS
    truck_route, truck_distance = greedy_large_neighborhood_search(distance_matrix, iterations=200, neighborhood_size=4, verbose=True)

    # Thêm drone vào tuyến đường
    truck_path, drone_path, vehicle_assignment, total_distance = add_drone_to_route(truck_route, distance_matrix, drone_probability=0.4)

    # Tính tổng quãng đường của truck
    truck_only_distance = calculate_truck_only_distance(truck_path, distance_matrix)

    # In kết quả
    print("\nOptimal route (starting from 0):", truck_path)
    print("Total distance (truck):", truck_only_distance)
    print("Vehicle assignment (1=truck, 2=drone):", vehicle_assignment)

if __name__ == "__main__":
    main()
