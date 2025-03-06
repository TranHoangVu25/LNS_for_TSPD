import numpy as np
import pandas as pd
import time
import random


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
            dist = np.sqrt((coordinates[i][1] - coordinates[j][1]) ** 2 +
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
            cs = coordinates[current_city][3] / visited_time_pre if visited_time_pre > 0 else coordinates[current_city][
                3]
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
# Sinh giải pháp ban đầu
# --------------------------

def generate_initial_particle(n_cities):
    """
    Sinh giải pháp ban đầu: permutation ngẫu nhiên của các thành phố (ngoại trừ depot),
    sau đó chèn depot vào đầu và cuối; ban đầu, gán tất cả là truck (vehicle = 1).
    """
    cities = list(np.random.permutation(n_cities - 1) + 1)
    particle = [(0, 1)]
    particle.extend([(city, 1) for city in cities])
    particle.append((0, 1))
    return particle


# --------------------------
# Cập nhật phân công xe (Truck/Drone)
# --------------------------

def dispatch_drone_by_ratio(particle, coordinates, distance_matrix, dispatch_ratio=0.3, drone_flight_limit=4):
    """
    Cập nhật phân công phương tiện cho particle:
      - Điểm đầu và cuối luôn là truck.
      - Với xác suất dispatch_ratio, kiểm tra xem tồn tại điểm C sao cho:
            distance(A, B) + distance(B, C) <= drone_flight_limit.
    """
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


# --------------------------
# Hàm greedy_removal: loại bỏ các node có saving cao
# --------------------------

def greedy_removal(city_tour, neighborhood_size, distance_matrix):
    """
    Tính “saving” khi loại bỏ từng node (ngoại trừ depot) và loại bỏ các node có saving cao nhất.
    """
    removal_candidates = []
    for i in range(1, len(city_tour) - 1):  # Loại bỏ depot cuối cùng
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


# --------------------------
# Hàm best_insertion: chèn lại các node theo phương pháp tham lam
# --------------------------

def best_insertion(removed_nodes, city_tour, distance_matrix):
    """
    Chèn lại các node bị loại bỏ theo phương pháp tham lam (best insertion).
    """
    for node in removed_nodes:
        best_cost = float("inf")
        best_index = -1
        for i in range(1, len(city_tour) + 1):
            prev_node = city_tour[i - 1]
            next_node = city_tour[i % len(city_tour)]
            insertion_cost = distance_matrix[prev_node][node] + distance_matrix[node][next_node] - \
                             distance_matrix[prev_node][next_node]
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_index = i
        city_tour.insert(best_index, node)
    return city_tour


# --------------------------
# Hàm random_removal_insertion: áp dụng LNS
# --------------------------

def random_removal_insertion(particle, coordinates, distance_matrix, neighborhood_size=4):
    """
    Áp dụng LNS bằng cách:
      1. Loại bỏ 'neighborhood_size' thành phố (ngoại trừ depot) theo phương pháp greedy_removal.
      2. Chèn lại các node đã loại bỏ theo phương pháp best insertion.
      3. Cập nhật phân công xe (Truck/Drone).
    """
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


# --------------------------
# Hàm local_search_2_opt: cải thiện giải pháp bằng 2-opt
# --------------------------

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


# --------------------------
# Hàm tìm kiếm giải pháp tổng thể
# --------------------------

def greedy_search_emission(coordinates, iterations=100, neighborhood_size=4, verbose=True):
    """
    Thực hiện tìm kiếm giải pháp với các bước:
      - Generation 0: Sinh giải pháp ban đầu.
      - Generation 1: Cải thiện bằng phương pháp "Random Removal & Reinsertion" (LNS).
      - Generations 2 -> iterations+1: Cải thiện bằng 2-opt.
    """
    n_cities = len(coordinates)
    records = []
    data_file = "new data TSPD.txt"
    EV, ED = 674.3, 14.4
    alpha, beta = 0.01, 0.99
    truck_speed, drone_speed, drone_flight_limit = 50, 43.2, 4

    # Tiền tính ma trận khoảng cách
    distance_matrix = compute_distance_matrix(coordinates)

    # Generation 0: Random initial
    start = time.time()
    particle0 = generate_initial_particle(n_cities)
    particle0 = dispatch_drone_by_ratio(particle0, coordinates, distance_matrix)
    fitness0, truck_dist0 = calculate_fitness(particle0, coordinates, distance_matrix)
    end = time.time()
    exec_time = end - start
    records.append({
        "Data File": data_file,
        "Running time": 0,
        "Algorithm": "random initial",
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
    for i in range(1,iterations-3):
    # Generation 1: Random Removal & Reinsertion
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

    # Generations 2 -> iterations+1: 2-opt Improvement
    for i in range(1,iterations-96):
    # if True:
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
            print(
                f"Iteration {i + 1:3d}: Using 2-opt, Fitness = {best_fitness:.4f}, Truck distance = {current_truck_distance:.4f}")

    return records, best_particle, best_fitness, calculate_fitness(best_particle, coordinates, distance_matrix)[1]


# --------------------------
# Hàm export_csv: lưu kết quả ra file CSV
# --------------------------

def export_csv(records, output_file="result.csv"):
    df = pd.DataFrame(records)
    df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
    print(f"\nResults exported to {output_file}")


# --------------------------
# Hàm main: chạy thuật toán và lưu kết quả
# --------------------------

def main():
    i = 0
    while i < 20:
        file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\LNS_for_TSPD\\new data\\200.txt"
        coordinates = load_coordinates(file_path)

        records, best_particle, best_fitness, truck_distance = greedy_search_emission(coordinates, iterations=100,
                                                                                      neighborhood_size=4, verbose=True)

        best_route = [int(city[0]) for city in best_particle]
        vehicles = [city[1] for city in best_particle]
        print("\nFinal Best Solution (lowest fitness):")
        print("Best route:", ", ".join(str(x) for x in best_route))
        print("Vehicles (1: Truck, 2: Drone):", ", ".join(str(x) for x in vehicles))
        print("Best fitness (emission-based): {:.4f}".format(best_fitness))
        print("Total truck distance: {:.4f}".format(truck_distance))

        export_csv(records, output_file="Result/output_greedy_200.csv")
        i += 1


if __name__ == "__main__":
    main()
