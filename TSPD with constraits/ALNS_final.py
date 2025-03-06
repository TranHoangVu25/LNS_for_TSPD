import copy
import numpy as np
import random
import math
import pandas as pd
import time

############################################################################
# 1. HÀM LOAD DATA: Tạo hoặc đọc ma trận khoảng cách + tọa độ
############################################################################
def load_data():
    """
    Đọc dữ liệu từ file (id, x, y, max_wait_time) và xây dựng ma trận khoảng cách.
    """
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\LNS_for_TSPD\\new data\\200.txt"
    data = np.loadtxt(file_path)  # Mỗi dòng: id, x, y, max_wait_time
    N = len(data)
    
    coordinates = []
    for i in range(N):
        coordinates.append((data[i, 0], data[i, 1], data[i, 2], int(data[i, 3])))
    
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i][j] = math.sqrt(
                    (coordinates[i][1] - coordinates[j][1])**2 +
                    (coordinates[i][2] - coordinates[j][2])**2
                )
    return coordinates, dist_matrix

############################################################################
# 2. HÀM TÍNH FITNESS
############################################################################
def calculate_fitness(particle, coordinates, distance_matrix,
                      EV=674.3, ED=14.4, alpha=0.01, beta=0.99):
    """
    particle: [(city_id, vehicle), ...] với vehicle = 1 (Truck) hoặc 2 (Drone)
    coordinates: [(id, x, y, max_wait_time), ...]
    distance_matrix: Ma trận khoảng cách giữa các thành phố.
    
    Trả về:
      - fitness_value: giá trị fitness (alpha * GHG + beta / CS)
      - truck_distance: tổng quãng đường Truck
      - drone_distance: tổng quãng đường Drone
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
    return fitness_value, truck_distance, drone_distance

############################################################################
# 3. SINH GIẢI PHÁP BAN ĐẦU
############################################################################
def generate_initial_particle(n_cities):
    """
    Sinh lời giải ban đầu: hoán vị ngẫu nhiên của các city (trừ depot),
    sau đó chèn depot vào đầu và cuối, gán mặc định tất cả là Truck.
    """
    cities = list(np.random.permutation(n_cities - 1) + 1)
    particle = [(0, 1)]
    particle.extend([(city, 1) for city in cities])
    particle.append((0, 1))
    return particle

############################################################################
# 4. CẬP NHẬT PHƯƠNG TIỆN (DISPATCH DRONE)
############################################################################
def dispatch_drone_by_ratio(particle, coordinates, distance_matrix,
                            dispatch_ratio=0.3, drone_flight_limit=4):
    """
    Cập nhật phân công phương tiện:
      - Điểm đầu và cuối luôn Truck (1).
      - Với xác suất dispatch_ratio, nếu điều kiện bay của Drone thỏa mãn,
        gán điểm đó là Drone (2).
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

############################################################################
# 5. OPERATOR REMOVAL & INSERTION
############################################################################
def random_removal(city_order, num_removals, distance_matrix):
    """
    Loại bỏ ngẫu nhiên num_removals điểm từ city_order (ngoại trừ depot ở đầu và cuối).
    Trả về: (danh sách các node bị loại, danh sách tour sau khi loại bỏ)
    """
    if len(city_order) <= 2:
        return [], city_order
    removed = set()
    while len(removed) < num_removals:
        candidate = random.choice(city_order[1:-1])
        removed.add(candidate)
    removed_nodes = list(removed)
    remaining_tour = [city for city in city_order if city not in removed_nodes]
    return removed_nodes, remaining_tour

def cheapest_insertion(removed_nodes, city_order, distance_matrix):
    """
    Chèn lại từng node trong removed_nodes vào city_order theo cách chèn rẻ nhất.
    Với mỗi node, tìm vị trí chèn sao cho chi phí chèn thấp nhất.
    """
    for node in removed_nodes:
        best_cost = float("inf")
        best_index = -1
        for i in range(1, len(city_order) + 1):
            prev_node = city_order[i - 1]
            next_node = city_order[i % len(city_order)]
            cost = (distance_matrix[prev_node][node] +
                    distance_matrix[node][next_node] -
                    distance_matrix[prev_node][next_node])
            if cost < best_cost:
                best_cost = cost
                best_index = i
        city_order.insert(best_index, node)
    return city_order

############################################################################
# 6. LOCAL SEARCH 2-OPT (TRÊN PARTICLE)
############################################################################
def local_search_2_opt(particle, coordinates, distance_matrix):
    """
    Cải thiện lời giải bằng phương pháp 2-opt:
      Duyệt qua các cặp chỉ số (i, j) trong particle (ngoại trừ depot),
      đảo ngược đoạn [i, j], cập nhật lại phân công Drone và tính lại fitness.
      Nếu lời giải mới tốt hơn, cập nhật best_particle.
    """
    best_particle = particle[:]
    best_fitness, _, _ = calculate_fitness(particle, coordinates, distance_matrix)
    n = len(particle)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_particle = particle[:]
            new_particle[i:j+1] = new_particle[i:j+1][::-1]
            new_particle = dispatch_drone_by_ratio(new_particle, coordinates, distance_matrix)
            new_fitness, _, _ = calculate_fitness(new_particle, coordinates, distance_matrix)
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_particle = new_particle[:]
    return best_particle, best_fitness

############################################################################
# 7. EXPORT CSV
############################################################################
def export_csv(records, output_file="Result/alns_results.csv"):
    df = pd.DataFrame(records)
    df.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
    print(f"\nResults exported to {output_file}")

############################################################################
# 8. ALNS CHÍNH (Adaptive Large Neighborhood Search)
# Trong 100 vòng: vòng 0 là khởi tạo, vòng 1-96 là ALNS, vòng 97-99 là 2-opt
############################################################################
def adaptive_large_neighborhood_search(coordinates, distance_matrix,
                                       iterations=100, removal_fraction=0.2,
                                       rho=0.1, local_search=True, verbose=True,
                                       output_file="Result/alns_results.csv"):
    n_cities = len(coordinates)
    
    # Vòng 0: Khởi tạo ngẫu nhiên
    initial_particle = generate_initial_particle(n_cities)
    initial_particle = dispatch_drone_by_ratio(initial_particle, coordinates, distance_matrix)
    best_particle = initial_particle
    best_fitness, best_truck_dist, best_drone_dist = calculate_fitness(best_particle, coordinates, distance_matrix)
    
    # Lưu lời giải ban đầu vào CSV (Iteration 0)
    records = []
    record = {
        "Iteration": 0,
        "Algorithm": "Initial Random",
        "Fitness": best_fitness,
        "Truck Distance": best_truck_dist,
        "Drone Distance": best_drone_dist,
        "Execution Time": 0,
        "Solution": str([int(city) for city, veh in best_particle]),
        "Truck Route": str([int(city) for city, veh in best_particle if veh == 1]),
        "Drone Route": str([int(city) for city, veh in best_particle if veh == 2])
    }
    records.append(record)
    export_csv([record], output_file=output_file)
    
    # Vòng 1 đến 96: ALNS (Removal & Insertion)
    for count in range(1, 97):
        start_time = time.time()
        if verbose:
            print(f"Iteration = {count}, Fitness = {best_fitness:.4f}")
        current_order = [city for city, veh in best_particle]
        removal_op = random.choices([random_removal], weights=[1.0])[0]
        insertion_op = random.choices([cheapest_insertion], weights=[1.0])[0]
        num_removals = int(removal_fraction * (n_cities - 1))
        
        removed_nodes, remaining_order = removal_op(current_order, num_removals, distance_matrix)
        new_order = insertion_op(removed_nodes, remaining_order, distance_matrix)
        if new_order[0] != 0:
            new_order.insert(0, 0)
        if new_order[-1] != 0:
            new_order.append(0)
        
        new_particle = [(city, 1) for city in new_order]
        new_particle = dispatch_drone_by_ratio(new_particle, coordinates, distance_matrix)
        new_fitness, new_truck_dist, new_drone_dist = calculate_fitness(new_particle, coordinates, distance_matrix)
        
        if new_fitness < best_fitness:
            best_particle = new_particle
            best_fitness = new_fitness
            best_truck_dist = new_truck_dist
            best_drone_dist = new_drone_dist
        
        end_time = time.time()
        exec_time = end_time - start_time
        
        record = {
            "Iteration": count,
            "Algorithm": "ALNS",
            "Fitness": best_fitness,
            "Truck Distance": best_truck_dist,
            "Drone Distance": best_drone_dist,
            "Execution Time": exec_time,
            "Solution": str([int(city) for city, veh in best_particle]),
            "Truck Route": str([int(city) for city, veh in best_particle if veh == 1]),
            "Drone Route": str([int(city) for city, veh in best_particle if veh == 2])
        }
        records.append(record)
        export_csv([record], output_file=output_file)
    
    # Vòng 97 đến 99: 2-opt Local Search
    for i in range(3):
        start_time = time.time()
        best_particle, best_fitness = local_search_2_opt(best_particle, coordinates, distance_matrix)
        best_fitness, best_truck_dist, best_drone_dist = calculate_fitness(best_particle, coordinates, distance_matrix)
        end_time = time.time()
        exec_time = end_time - start_time
        iteration_label = 96 + i + 1  # Iteration 97, 98, 99
        
        record = {
            "Iteration": iteration_label,
            "Algorithm": "Local Search 2-opt",
            "Fitness": best_fitness,
            "Truck Distance": best_truck_dist,
            "Drone Distance": best_drone_dist,
            "Execution Time": exec_time,
            "Solution": str([int(city) for city, veh in best_particle]),
            "Truck Route": str([int(city) for city, veh in best_particle if veh == 1]),
            "Drone Route": str([int(city) for city, veh in best_particle if veh == 2])
        }
        records.append(record)
        export_csv([record], output_file=output_file)
        
        # In kết quả 2-opt cho vòng hiện tại
        print(f"2-opt Iteration {iteration_label}: Fitness = {best_fitness:.4f}")

    final_route = [int(city) + 1 for city, veh in best_particle]
    return best_particle, final_route, best_fitness, best_truck_dist, best_drone_dist, records

############################################################################
# 9. MAIN
############################################################################
if __name__ == '__main__':
    i = 0
    while i < 20:
        coordinates, dist_matrix = load_data()
        best_particle, route, fitness, truck_dist, drone_dist, records = adaptive_large_neighborhood_search(
            coordinates,
            dist_matrix,
            iterations=100,       # Tổng 100 vòng: 1 khởi tạo + 96 ALNS + 3 2-opt
            removal_fraction=0.2,
            rho=0.1,
            local_search=True,
            verbose=True,
            output_file="Result/alns_200.csv"
        )

        print("\nOptimal Route (1-index):", route)
        print(f"Optimal Fitness (emission-based): {fitness:.4f}")
        print(f"Total Truck Distance: {truck_dist:.4f}")
        print(f"Total Drone Distance: {drone_dist:.4f}")

        drone_route = [city + 1 for city, veh in best_particle if veh == 2]
        print("Drone serves these cities (1-index):", list(map(int, drone_route)))
        i += 1
