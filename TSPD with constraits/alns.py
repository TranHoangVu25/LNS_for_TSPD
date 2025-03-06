import copy
import numpy as np
import random
import math

############################################################################
# 1. HÀM LOAD DATA: Tạo hoặc đọc ma trận khoảng cách + toạ độ
############################################################################
def load_data():
    """
    Ví dụ: Đọc dữ liệu từ file (id, x, y, max_wait_time).
    Sau đó, xây dựng ma trận khoảng cách (dist_matrix).
    Bạn có thể thay đổi đường dẫn file và cấu trúc tuỳ theo nhu cầu.
    """
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\LNS_for_TSPD\\new data\\50.txt"

    data = np.loadtxt(file_path)  # Mỗi dòng: id, x, y, max_wait_time
    N = len(data)

    # Lưu lại toạ độ (bao gồm max_wait_time)
    coordinates = []
    for i in range(N):
        coordinates.append((data[i, 0], data[i, 1], data[i, 2], int(data[i, 3])))

    # Tạo ma trận khoảng cách
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
# Trả về (fitness_value, truck_distance, drone_distance)
############################################################################
def calculate_fitness(particle, coordinates, distance_matrix,
                      EV=674.3, ED=14.4, alpha=0.01, beta=0.99):
    """
    particle: [(city_id, vehicle), ...]  với vehicle = 1 (Truck) hoặc 2 (Drone)
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
            # Tính thời gian di chuyển
            if particle[i - 1][1] == 1:
                seg_time = distance_matrix[last_truck][current_city] / truck_speed
                visited_time_pre = seg_time
            else:
                x = distance_matrix[branch_start][current_city] / truck_speed
                waiting_time = max(x, cumulative_time)
                seg_time = distance_matrix[last_truck][current_city] / truck_speed
                visited_time_pre = waiting_time + seg_time

            # Tính Customer Satisfaction
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
    Tạo thứ tự ngẫu nhiên các city (trừ city 0),
    thêm city 0 vào đầu/cuối và gán Truck = 1.
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
    - Điểm đầu và cuối luôn Truck (1).
    - Với xác suất dispatch_ratio, nếu distance(A,B)+distance(B,C) <= drone_flight_limit
      thì gán điểm B là Drone (2).
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
# 5. HÀM LOẠI BỎ (greedy_removal) & CHÈN (best_insertion)
############################################################################
def greedy_removal(city_order, neighborhood_size, distance_matrix):
    removal_candidates = []
    for i in range(1, len(city_order) - 1):
        prev_index = i - 1
        next_index = i + 1
        saving = (distance_matrix[city_order[prev_index]][city_order[i]] +
                  distance_matrix[city_order[i]][city_order[next_index]] -
                  distance_matrix[city_order[prev_index]][city_order[next_index]])
        removal_candidates.append((city_order[i], saving))
    removal_candidates.sort(key=lambda x: x[1], reverse=True)
    removed_nodes = [node for node, _ in removal_candidates[:neighborhood_size]]
    remaining_tour = [city for city in city_order if city not in removed_nodes]
    return removed_nodes, remaining_tour

def best_insertion(removed_nodes, city_order, distance_matrix):
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
def local_search_2_opt_fitness(particle, coordinates, distance_matrix,
                               iterations=100, verbose=True):
    best_particle = particle[:]
    best_fitness, _, _ = calculate_fitness(best_particle, coordinates, distance_matrix)
    n = len(particle)
    improved = True
    iter_count = 0
    while improved and iter_count < iterations:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_particle = best_particle.copy()
                # Đảo ngược đoạn [i:j+1]
                new_particle[i:j+1] = list(reversed(new_particle[i:j+1]))
                # Cập nhật Drone
                new_particle = dispatch_drone_by_ratio(new_particle, coordinates, distance_matrix)
                new_fitness, _, _ = calculate_fitness(new_particle, coordinates, distance_matrix)
                if new_fitness < best_fitness:
                    best_particle = new_particle
                    best_fitness = new_fitness
                    improved = True
        iter_count += 1
        if verbose:
            print(f"Local search iteration {iter_count}: Fitness = {best_fitness:.4f}")
    return best_particle, best_fitness

############################################################################
# 7. ALNS CHÍNH
############################################################################
def adaptive_large_neighborhood_search(coordinates, distance_matrix,
                                       iterations=100, removal_fraction=0.2,
                                       rho=0.1, local_search=True, verbose=True):
    n_cities = len(coordinates)
    # Tạo giải pháp ban đầu
    initial_particle = generate_initial_particle(n_cities)
    # Cập nhật Drone
    initial_particle = dispatch_drone_by_ratio(initial_particle, coordinates, distance_matrix)
    best_particle = initial_particle
    best_fitness, best_truck_dist, best_drone_dist = calculate_fitness(best_particle, coordinates, distance_matrix)
    
    # Định nghĩa operator
    removal_ops = [greedy_removal]
    insertion_ops = [best_insertion]
    weights_removal = [1.0]
    weights_insertion = [1.0]
    
    count = 0
    while count <= iterations:
        if verbose and count > 0:
            print(f"Iteration = {count}, Fitness = {best_fitness:.4f}")
        # city_order (0-index)
        current_order = [city for city, veh in best_particle]
        removal_op = random.choices(removal_ops, weights=weights_removal)[0]
        insertion_op = random.choices(insertion_ops, weights=weights_insertion)[0]
        num_removals = int(removal_fraction * (n_cities - 1))

        # Loại bỏ
        removed_nodes, remaining_order = removal_op(current_order, num_removals, distance_matrix)
        # Chèn lại
        new_order = insertion_op(removed_nodes, remaining_order, distance_matrix)
        # Đảm bảo depot
        if new_order[0] != 0:
            new_order.insert(0, 0)
        if new_order[-1] != 0:
            new_order.append(0)

        # Chuyển thành particle
        new_particle = [(city, 1) for city in new_order]
        # Cập nhật Drone
        new_particle = dispatch_drone_by_ratio(new_particle, coordinates, distance_matrix)
        new_fitness, _, _ = calculate_fitness(new_particle, coordinates, distance_matrix)
        
        # So sánh
        if new_fitness < best_fitness:
            best_particle = new_particle
            best_fitness = new_fitness
            weights_removal[0] *= (1 + rho)
            weights_insertion[0] *= (1 + rho)
        else:
            weights_removal[0] *= (1 - rho)
            weights_insertion[0] *= (1 - rho)
        
        # Chuẩn hóa trọng số
        total_w_rem = sum(weights_removal)
        total_w_ins = sum(weights_insertion)
        weights_removal = [w / total_w_rem for w in weights_removal]
        weights_insertion = [w / total_w_ins for w in weights_insertion]
        
        count += 1

    # Local search
    if local_search:
        best_particle, best_fitness = local_search_2_opt_fitness(
            best_particle, coordinates, distance_matrix,
            iterations=100, verbose=verbose
        )
        best_fitness, best_truck_dist, best_drone_dist = calculate_fitness(best_particle, coordinates, distance_matrix)
    
    # Lộ trình dạng 1-index
    final_route = [int(city)+1 for city, veh in best_particle]
    return best_particle, final_route, best_fitness, best_truck_dist, best_drone_dist

############################################################################
# 8. MAIN
############################################################################
if __name__ == '__main__':
    coordinates, dist_matrix = load_data()

    best_particle, route, fitness, truck_dist, drone_dist = adaptive_large_neighborhood_search(
        coordinates,
        dist_matrix,
        iterations=100,
        removal_fraction=0.2,
        rho=0.1,
        local_search=True,
        verbose=True
    )

    # In kết quả
    print("\nOptimal Route (1-index):", route)
    print(f"Optimal Fitness (emission-based): {fitness:.4f}")
    print(f"Total Truck Distance: {truck_dist:.4f}")
    print(f"Total Drone Distance: {drone_dist:.4f}")

    # In chi tiết lộ trình Drone
    # Lấy danh sách các thành phố được phục vụ bởi Drone
    drone_route = [city+1 for (city, veh) in best_particle if veh == 2]
    print("Drone serves these cities (1-index):", list(map(int, drone_route)))
