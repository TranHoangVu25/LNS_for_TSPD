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
            # Giả sử file có: city_id, x, y, max_wait_time
            coordinates.append((float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])))
    return coordinates


def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)


# --------------------------
# Hàm tính fitness
# --------------------------

def calculate_fitness(particle, coordinates,
                      EV=674.3, ED=14.4,  # Hệ số khí thải của truck và drone
                      alpha=0.01, beta=0.99):  # Trọng số cho khí thải và độ hài lòng khách hàng
    """
    Tính fitness của một giải pháp (particle) dựa trên:
      1. Khí thải (GHG):
           - Truck: tổng quãng đường truck × EV.
           - Drone: tổng quãng đường drone × ED.
      2. Customer Satisfaction (CS):
           Với mỗi điểm i, CS(i) = (thời gian chờ của khách hàng tại i) / (visited_time tại i),
           với visited_time được tính theo 3 trường hợp:
             - Trường hợp 1: Truck chạy trực tiếp.
             - Trường hợp 2: Điểm được phục vụ bởi drone.
             - Trường hợp 3: Truck phục vụ ngay sau một chuỗi drone.

    Fitness = alpha * (total GHG) + beta / (sum CS)
    Mục tiêu: tối thiểu hóa fitness.
    """
    truck_speed = 50
    drone_speed = 43.2

    total_cs = 0.0
    truck_distance = 0.0
    drone_distance = 0.0

    # Khởi tạo: điểm đầu tiên (depot) phục vụ bởi truck
    last_truck = coordinates[int(particle[0][0])]
    branch_start = last_truck  # điểm truck gốc của nhánh drone (nếu có)
    cumulative_time = 0.0  # tích lũy thời gian của nhánh drone
    visited_time_pre = 0.0  # thời gian đã hiệu chỉnh (cho CS sau nhánh drone)

    for i in range(1, len(particle)):
        current_city = coordinates[int(particle[i][0])]
        if particle[i][1] == 1:  # Nếu điểm i được phục vụ bởi truck
            if particle[i - 1][1] == 1:
                # Trường hợp 1: Truck chạy trực tiếp
                seg_time = euclidean_distance(last_truck, current_city) / truck_speed
                visited_time_pre = seg_time  # reset visited_time
            else:
                # Trường hợp 3: Truck phục vụ ngay sau chuỗi drone
                x = euclidean_distance(branch_start, current_city) / truck_speed
                waiting_time = max(x, cumulative_time)
                visited_time_pre += waiting_time + (euclidean_distance(last_truck, current_city) / truck_speed)
            cs = current_city[3] / visited_time_pre if visited_time_pre > 0 else current_city[3]
            total_cs += cs
            truck_distance += euclidean_distance(last_truck, current_city)
            last_truck = current_city
            branch_start = current_city  # reset branch_start
            cumulative_time = 0.0  # reset cumulative_time
        else:
            # Trường hợp 2: Điểm được phục vụ bởi drone
            seg_time = euclidean_distance(last_truck, current_city) / drone_speed
            if i == 1 or particle[i - 1][1] == 1:
                cumulative_time = seg_time
                branch_start = last_truck
            else:
                cumulative_time += seg_time
            cs = current_city[3] / seg_time if seg_time > 0 else current_city[3]
            total_cs += cs
            drone_distance += euclidean_distance(last_truck, current_city)
            # last_truck không thay đổi
    ghg_truck = truck_distance * EV
    ghg_drone = drone_distance * ED
    total_ghg = ghg_truck + ghg_drone
    fitness_value = alpha * total_ghg + beta / total_cs
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

def dispatch_drone_by_ratio(particle, coordinates, dispatch_ratio=0.3, drone_flight_limit=4):
    """
    Cập nhật phân công phương tiện cho particle:
      - Điểm đầu và cuối luôn là truck.
      - Nếu điểm liền trước đã dùng drone, thì điểm hiện tại buộc phải là truck.
      - Với xác suất dispatch_ratio, kiểm tra xem có tồn tại điểm C (trong các thành phố giữa) sao cho:
            euclidean_distance(A, B) + euclidean_distance(B, C) <= drone_flight_limit.
        Nếu có, gán điểm hiện tại là drone (2), ngược lại gán truck (1).
    """
    n = len(particle)
    updated = [(particle[0][0], 1)]
    for i in range(1, n - 1):
        city_id = particle[i][0]
        if updated[-1][1] == 2:
            updated.append((city_id, 1))
            continue
        if np.random.rand() < dispatch_ratio:
            A = coordinates[int(particle[i - 1][0])]
            B = coordinates[int(city_id)]
            candidate_found = False
            for k in range(i + 1, n - 1):
                C = coordinates[int(particle[k][0])]
                if euclidean_distance(A, B) + euclidean_distance(B, C) <= drone_flight_limit:
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
# LNS với Random Removal (không dùng greedy removal)
# --------------------------

def random_removal(city_tour, neighborhood_size):
    """
    Loại bỏ ngẫu nhiên 'neighborhood_size' thành phố từ city_tour (không loại bỏ depot đầu).
    Trả về: (danh sách các node bị loại, danh sách tour sau khi loại bỏ)
    """
    removed = random.sample(city_tour[1:], neighborhood_size)
    remaining = [t for t in city_tour if t not in removed]
    return removed, remaining


def best_insertion(removed_nodes, city_tour, distance_matrix):
    """
    Chèn lại các thành phố bị loại vào city_tour theo phương pháp best insertion.
    Với mỗi node bị loại, thử chèn vào tất cả các vị trí và chọn vị trí có chi phí chèn nhỏ nhất.
    """
    for node in removed_nodes:
        best_cost = float('inf')
        best_index = -1
        # Thử chèn từ vị trí 1 đến len(city_tour)
        for i in range(1, len(city_tour) + 1):
            last_node = city_tour[i - 1]
            next_node = city_tour[i % len(city_tour)]
            cost = distance_matrix[last_node, node] + distance_matrix[node, next_node] - distance_matrix[
                last_node, next_node]
            if cost < best_cost:
                best_cost = cost
                best_index = i
        city_tour.insert(best_index, node)
    return city_tour


def random_removal_insertion(particle, coordinates, neighborhood_size=4):
    """
    Áp dụng LNS bằng cách:
      1. Loại bỏ ngẫu nhiên 'neighborhood_size' thành phố (ngoại trừ depot) từ particle.
      2. Chèn lại các thành phố bị loại theo best insertion.
      3. Cập nhật phân công xe (Truck/Drone).
    Trả về particle mới.
    """
    # Chuyển particle thành danh sách các city_id
    city_tour = [int(city[0]) for city in particle]
    removed_nodes, remaining_tour = random_removal(city_tour, neighborhood_size)

    # Tính ma trận khoảng cách
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = euclidean_distance(coordinates[i], coordinates[j])

    new_tour = best_insertion(removed_nodes, remaining_tour, distance_matrix)

    # Chuyển lại thành particle với tất cả gán xe tải ban đầu
    new_particle = [(city, 1) for city in new_tour]
    # Đảm bảo depot xuất hiện ở đầu và cuối
    if new_particle[0][0] != 0:
        new_particle.insert(0, (0, 1))
    if new_particle[-1][0] != 0:
        new_particle.append((0, 1))

    # Cập nhật phân công xe (Truck/Drone)
    new_particle = dispatch_drone_by_ratio(new_particle, coordinates)
    return new_particle


# --------------------------
# Local Search 2-opt
# --------------------------

def local_search_2_opt(particle, coordinates):
    best_particle = particle[:]
    best_fitness, _ = calculate_fitness(particle, coordinates)
    n = len(particle)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_particle = particle[:]
            new_particle[i:j + 1] = new_particle[i:j + 1][::-1]
            new_particle = dispatch_drone_by_ratio(new_particle, coordinates)
            new_fitness, _ = calculate_fitness(new_particle, coordinates)
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_particle = new_particle[:]
    return best_particle, best_fitness


# --------------------------
# Tìm kiếm giải pháp tổng thể (LNS + 2-opt)
# --------------------------

def lns_search_emission(coordinates, iterations=80, neighborhood_size=4, verbose=True):
    """
    Thực hiện tìm kiếm giải pháp với các bước:
      - Generation 0: Random initial.
      - Generation 1: Cải thiện bằng LNS (Random Removal & Best Insertion) với neighborhood_size.
      - Generations 2 -> iterations+1: Cải thiện bằng 2-opt.
    Trả về (records, best_particle, best_fitness, truck_distance).
    """
    n_cities = len(coordinates)
    records = []
    data_file = "new data TSPD.txt"
    EV, ED = 674.3, 14.4
    alpha, beta = 0.01, 0.99
    truck_speed, drone_speed, drone_flight_limit = 50, 43.2, 4

    # --- Generation 0: Random initial ---
    start = time.time()
    particle0 = generate_initial_particle(n_cities)
    particle0 = dispatch_drone_by_ratio(particle0, coordinates)
    fitness0, truck_dist0 = calculate_fitness(particle0, coordinates)
    end = time.time()
    exec_time = end - start
    records.append({
        "Data File": data_file,
        "Algorithm": "random initial",
        "generation": 0,
        "Fitness": fitness0,
        "executing time": exec_time,
        "solution": str([int(city[0]) for city in particle0]),
        "truck route": str([int(city[0]) for city in particle0 if city[1] == 1]),
        "drone route": str([int(city[0]) for city in particle0 if city[1] == 2])
    })

    best_particle = particle0
    best_fitness = fitness0

    # --- Generation 1: LNS (Random Removal & Insertion) ---
    start = time.time()
    particle1 = random_removal_insertion(best_particle, coordinates, neighborhood_size)
    fitness1, _ = calculate_fitness(particle1, coordinates)
    end = time.time()
    exec_time = end - start
    if fitness1 < best_fitness:
        best_particle = particle1
        best_fitness = fitness1
    records.append({
        "Data File": data_file,
        "Algorithm": "LNS (Random Removal & Insertion)",
        "generation": 1,
        "Fitness": best_fitness,
        "executing time": exec_time,
        "solution": str([int(city[0]) for city in best_particle]),
        "truck route": str([int(city[0]) for city in best_particle if city[1] == 1]),
        "drone route": str([int(city[0]) for city in best_particle if city[1] == 2])
    })

    # --- Generations 2 -> iterations+1: 2-opt Improvement ---
    for it in range(iterations):
        start = time.time()
        new_particle, new_fitness = local_search_2_opt(best_particle, coordinates)
        end = time.time()
        exec_time = end - start
        if new_fitness < best_fitness:
            best_particle = new_particle
            best_fitness = new_fitness
        records.append({
            "Data File": data_file,
            "Algorithm": "2-opt",
            "generation": it + 2,
            "Fitness": best_fitness,
            "executing time": exec_time,
            "solution": str([int(city[0]) for city in best_particle]),
            "truck route": str([int(city[0]) for city in best_particle if city[1] == 1]),
            "drone route": str([int(city[0]) for city in best_particle if city[1] == 2])
        })
        if verbose:
            current_truck_distance = calculate_fitness(best_particle, coordinates)[1]
            print(
                f"Iteration {it + 1:3d}: 2-opt, Fitness = {best_fitness:.4f}, Truck distance = {current_truck_distance:.4f}")

    return records, best_particle, best_fitness, calculate_fitness(best_particle, coordinates)[1]


# --------------------------
# Xuất kết quả ra file CSV
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
    while i < 21:
        # Chọn file dữ liệu (ví dụ với 100 thành phố)
        file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\LNS_for_TSPD\\new data\\100.txt"
        coordinates = load_coordinates(file_path)

        records, best_particle, best_fitness, truck_distance = lns_search_emission(coordinates, iterations=100,
                                                                                   neighborhood_size=4, verbose=True)

        best_route = [int(city[0]) for city in best_particle]
        vehicles = [city[1] for city in best_particle]
        print("\nFinal Best Solution (lowest fitness):")
        print("Best route:", ", ".join(str(x) for x in best_route))
        print("Vehicles (1: Truck, 2: Drone):", ", ".join(str(x) for x in vehicles))
        print("Best fitness (emission-based): {:.4f}".format(best_fitness))
        print("Total truck distance: {:.4f}".format(truck_distance))

        export_csv(records, output_file="Result/output_100.csv")
        i += 1


if __name__ == "__main__":
    main()
