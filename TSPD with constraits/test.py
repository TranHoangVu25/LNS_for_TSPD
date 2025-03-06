import numpy as np
import pandas as pd
import time
import random
import math

# --------------------------
# Hàm hỗ trợ: đọc dữ liệu tọa độ
# --------------------------

def load_coordinates(file_path):
    """
    Đọc dữ liệu tọa độ từ file.
    Mỗi dòng có dạng: [city_id, x, y, max_wait_time]
    Các giá trị được cách nhau bởi khoảng trắng.
    """
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Giả sử file chứa city_id, x, y, max_wait_time
            # Nếu không cần city_id, bạn có thể bỏ qua và dùng chỉ số dòng làm city_id.
            coordinates.append((float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])))
    return coordinates

# --------------------------
# Hàm tiền tính ma trận khoảng cách
# --------------------------

def compute_distance_matrix(coordinates):
    """
    Tạo ma trận khoảng cách để tránh tính toán lại nhiều lần.
    """
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((coordinates[i][1] - coordinates[j][1]) ** 2 +
                             (coordinates[i][2] - coordinates[j][2]) ** 2)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix

# --------------------------
# Hàm tính fitness
# --------------------------

def calculate_fitness(particle, coordinates, distance_matrix,
                      EV=674.3, ED=14.4, alpha=0.01, beta=0.99):
    """
    Tính fitness của một giải pháp (particle) dựa trên:
      - Khí thải (GHG) của Truck và Drone.
      - Customer Satisfaction (CS).
    Mục tiêu: tối thiểu hóa fitness.
    
    Particle: danh sách các tuple (city, mode)
      - mode = 1: Truck, mode = 2: Drone
    """
    truck_speed = 50
    drone_speed = 43.2

    total_cs = 0.0
    truck_distance = 0.0
    drone_distance = 0.0

    last_truck = int(particle[0][0])
    branch_start = last_truck
    cumulative_time = 0.0
    visited_time_pre = 0.0

    for i in range(1, len(particle)):
        current_city = int(particle[i][0])
        if particle[i][1] == 1:  # Truck
            if particle[i - 1][1] == 1:
                seg_time = distance_matrix[last_truck][current_city] / truck_speed
                visited_time_pre = seg_time
            else:
                x = distance_matrix[branch_start][current_city] / truck_speed
                waiting_time = max(x, cumulative_time)
                seg_time = distance_matrix[last_truck][current_city] / truck_speed
                visited_time_pre = waiting_time + seg_time
            # Tính Customer Satisfaction: dùng max_wait_time (tại vị trí current_city)
            cs = coordinates[current_city][3] / visited_time_pre if visited_time_pre > 0 else coordinates[current_city][3]
            total_cs += cs
            truck_distance += distance_matrix[last_truck][current_city]
            last_truck = current_city
            branch_start = current_city
            cumulative_time = 0.0
        else:  # Drone
            seg_time = distance_matrix[last_truck][current_city] / drone_speed
            if i == 1 or particle[i - 1][1] == 1:
                cumulative_time = seg_time
                branch_start = last_truck
            else:
                cumulative_time += seg_time
            cs = coordinates[current_city][3] / seg_time if seg_time > 0 else coordinates[current_city][3]
            total_cs += cs
            drone_distance += distance_matrix[last_truck][current_city]

    total_ghg = (truck_distance * EV) + (drone_distance * ED)
    fitness_value = alpha * total_ghg + beta / max(total_cs, 1e-6)
    return fitness_value, truck_distance

# --------------------------
# Hàm khởi tạo giải pháp ban đầu từ file kết quả
# --------------------------

def generate_initial_particle_from_file(result_file_path):
    """
    Đọc file kết quả có định dạng:
      Best route: 0 5 8 14 17 7 12 6 ... 0
      Vehicles: 1 2 1 1 1 1 1 1 ... 1
      Best fitness: ...
      Total truck distance: ...
    
    Hàm sẽ lấy thông tin Best route và Vehicles để tạo particle dưới dạng:
        [(city, vehicle), (city, vehicle), ...]
    """
    best_route = None
    vehicles = None
    with open(result_file_path, "r") as f:
        for line in f:
            if line.startswith("Best route:"):
                parts = line.replace("Best route:", "").strip().split()
                best_route = [int(x) for x in parts]
            elif line.startswith("Vehicles:"):
                parts = line.replace("Vehicles:", "").strip().split()
                vehicles = [int(x) for x in parts]
    if best_route is None or vehicles is None:
        raise ValueError("File result không chứa thông tin 'Best route:' hoặc 'Vehicles:'.")
    if len(best_route) != len(vehicles):
        raise ValueError("Số lượng thành phố không khớp với số lượng vehicles.")
    particle = [(city, vehicle) for city, vehicle in zip(best_route, vehicles)]
    return particle

# --------------------------
# Các hàm cải thiện giải pháp (đã giữ nguyên cấu trúc ban đầu)
# --------------------------

def dispatch_drone_by_ratio(particle, coordinates, distance_matrix, dispatch_ratio=0.3, drone_flight_limit=4):
    n = len(particle)
    updated = [(particle[0][0], 1)]
    for i in range(1, n - 1):
        city_id = particle[i][0]
        if updated[-1][1] == 2:
            updated.append((city_id, 1))
            continue
        if np.random.rand() < dispatch_ratio:
            A = int(updated[-1][0])
            B = int(city_id)
            candidate_found = False
            for k in range(i + 1, n - 1):
                C = int(particle[k][0])
                if distance_matrix[A][B] + distance_matrix[B][C] <= drone_flight_limit:
                    candidate_found = True
                    break
            if candidate_found:
                updated.append((city_id, 2))
            else:
                updated.append((city_id, 1))
        else:
            updated.append((city_id, 1))
    updated.append((particle[-1][0], 1))
    return updated

def greedy_removal(city_tour, neighborhood_size, distance_matrix):
    removal_candidates = []
    for i in range(1, len(city_tour) - 1):
        prev_index = i - 1
        next_index = i + 1
        saving = distance_matrix[city_tour[prev_index]][city_tour[i]] + \
                 distance_matrix[city_tour[i]][city_tour[next_index]] - \
                 distance_matrix[city_tour[prev_index]][city_tour[next_index]]
        removal_candidates.append((city_tour[i], saving))
    removal_candidates.sort(key=lambda x: x[1], reverse=True)
    removed_nodes = [node for node, _ in removal_candidates[:neighborhood_size]]
    remaining_tour = [city for city in city_tour if city not in removed_nodes]
    return removed_nodes, remaining_tour

def best_insertion(removed_nodes, city_tour, distance_matrix):
    for node in removed_nodes:
        best_cost = float("inf")
        best_index = -1
        for i in range(1, len(city_tour) + 1):
            prev_node = city_tour[i - 1]
            next_node = city_tour[i % len(city_tour)]
            insertion_cost = distance_matrix[prev_node][node] + distance_matrix[node][next_node] - distance_matrix[prev_node][next_node]
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_index = i
        city_tour.insert(best_index, node)
    return city_tour

def random_removal_insertion(particle, coordinates, distance_matrix, neighborhood_size=4):
    city_tour = [int(city[0]) for city in particle]
    removed_nodes, remaining_tour = greedy_removal(city_tour, neighborhood_size, distance_matrix)
    new_tour = best_insertion(removed_nodes, remaining_tour, distance_matrix)
    if new_tour[0] != 0:
        new_tour.insert(0, 0)
    if new_tour[-1] != 0:
        new_tour.append(0)
    new_particle = [(city, 1) for city in new_tour]
    new_particle = dispatch_drone_by_ratio(new_particle, coordinates, distance_matrix)
    return new_particle

def local_search_2_opt(particle, coordinates, distance_matrix):
    best_particle = particle[:]
    best_fitness, _ = calculate_fitness(particle, coordinates, distance_matrix)
    n = len(particle)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_particle = particle[:]
            new_particle[i:j + 1] = new_particle[i:j + 1][::-1]
            new_particle = dispatch_drone_by_ratio(new_particle, coordinates, distance_matrix)
            new_fitness, _ = calculate_fitness(new_particle, coordinates, distance_matrix)
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_particle = new_particle[:]
    return best_particle, best_fitness

def greedy_search_emission(coordinates, iterations=100, neighborhood_size=4, verbose=True, initial_particle=None):
    """
    Tìm kiếm giải pháp tổng thể.
    Nếu initial_particle không None, sử dụng nó làm giải pháp ban đầu,
    ngược lại sinh ngẫu nhiên.
    """
    n_cities = len(coordinates)
    records = []
    data_file = "new data TSPD.txt"
    EV, ED = 674.3, 14.4
    alpha, beta = 0.01, 0.99
    truck_speed, drone_speed, drone_flight_limit = 50, 43.2, 4

    distance_matrix = compute_distance_matrix(coordinates)

    # Generation 0: Sử dụng initial_particle nếu có, ngược lại sinh ngẫu nhiên.
    start = time.time()
    if initial_particle is None:
        particle0 = generate_initial_particle_from_file(n_cities)
    else:
        particle0 = initial_particle
    particle0 = dispatch_drone_by_ratio(particle0, coordinates, distance_matrix)
    fitness0, truck_dist0 = calculate_fitness(particle0, coordinates, distance_matrix)
    end = time.time()
    exec_time = end - start
    records.append({
        "Data File": data_file,
        "Running time": 0,
        "Algorithm": "random initial / from file",
        "EV": EV,
        "ED": ED,
        "alpha": alpha,
        "beta": beta,
        "truck_speed": truck_speed,
        "drone_speed": drone_speed,
        "drone_flight_limit": drone_flight_limit,
        "generation": 0,
        "Fitness": fitness0,
        "excuting time": exec_time,
        "solution": str([int(city[0]) for city in particle0]),
        "truck route": str([int(city[0]) for city in particle0 if city[1] == 1]),
        "drone route": str([int(city[0]) for city in particle0 if city[1] == 2])
    })

    best_particle = particle0
    best_fitness = fitness0
    for i in range(1, iterations-3):
        start = time.time()
        particle1 = random_removal_insertion(best_particle, coordinates, distance_matrix, neighborhood_size)
        fitness1, _ = calculate_fitness(particle1, coordinates, distance_matrix)
        end = time.time()
        exec_time = end - start
        if fitness1 < best_fitness:
            best_particle = particle1
            best_fitness = fitness1
        records.append({
            "Data File": data_file,
            "Running time": i+1,
            "Algorithm": "Random Removal & Reinsertion",
            "EV": EV,
            "ED": ED,
            "alpha": alpha,
            "beta": beta,
            "truck_speed": truck_speed,
            "drone_speed": drone_speed,
            "drone_flight_limit": drone_flight_limit,
            "generation": i+1,
            "Fitness": best_fitness,
            "excuting time": exec_time,
            "solution": str([int(city[0]) for city in best_particle]),
            "truck route": str([int(city[0]) for city in best_particle if city[1] == 1]),
            "drone route": str([int(city[0]) for city in best_particle if city[1] == 2])
        })

    for i in range(1, iterations-96):
        start = time.time()
        new_particle, new_fitness = local_search_2_opt(best_particle, coordinates, distance_matrix)
        end = time.time()
        exec_time = end - start
        if new_fitness < best_fitness:
            best_particle = new_particle
            best_fitness = new_fitness
        records.append({
            "Data File": data_file,
            "Running time": i+86,
            "Algorithm": "2-opt",
            "EV": EV,
            "ED": ED,
            "alpha": alpha,
            "beta": beta,
            "truck_speed": truck_speed,
            "drone_speed": drone_speed,
            "drone_flight_limit": drone_flight_limit,
            "generation": i + 95,
            "Fitness": best_fitness,
            "excuting time": exec_time,
            "solution": str([int(city[0]) for city in best_particle]),
            "truck route": str([int(city[0]) for city in best_particle if city[1] == 1]),
            "drone route": str([int(city[0]) for city in best_particle if city[1] == 2])
        })
        if verbose:
            current_truck_distance = calculate_fitness(best_particle, coordinates, distance_matrix)[1]
            print(f"Iteration {i + 1:3d}: Using 2-opt, Fitness = {best_fitness:.4f}, Truck distance = {current_truck_distance:.4f}")

    return records, best_particle, best_fitness, calculate_fitness(best_particle, coordinates, distance_matrix)[1]

def export_csv(records, output_file="result.csv"):
    df = pd.DataFrame(records)
    df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
    print(f"\nResults exported to {output_file}")

# --------------------------
# Hàm main: chạy thuật toán và lưu kết quả
# --------------------------

def main():
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\LNS_for_TSPD\\new data\\50.txt"
    # Đường dẫn tới file kết quả đã lưu từ PSO (chứa "Best route:" và "Vehicles:")
    result_file = "Result/PSO/PSO.txt"
    
    # Sinh giải pháp ban đầu từ file kết quả
    initial_particle = generate_initial_particle_from_file(result_file)
    
    # Chạy thuật toán tìm kiếm cải thiện giải pháp dựa trên emission (sử dụng initial_particle)
    records, best_particle, best_fitness, truck_distance = greedy_search_emission(
        load_coordinates(file_path), iterations=100, neighborhood_size=4, verbose=True, initial_particle=initial_particle)
    
    best_route = [int(city[0]) for city in best_particle]
    vehicles = [int(city[1]) for city in best_particle]
    
    print("\nFinal Best Solution (lowest fitness):")
    print("Best route:", ", ".join(str(x) for x in best_route))
    print("Vehicles (1: Truck, 2: Drone):", ", ".join(str(x) for x in vehicles))
    print("Best fitness (emission-based): {:.4f}".format(best_fitness))
    print("Total truck distance: {:.4f}".format(truck_distance))
    
    # export_csv(records, output_file="Result/output_greedy_50.csv")

if __name__ == "__main__":
    main()
