"""
- This script is updated with emission fitness function and some constraints between truck and drone such as:
    + Limit distance of drone = 4.
    + Incorporate a waiting time ratio between truck and drone.
    + Drone is dispatched and identifies a return location using the formula:
         distance(i-1,i) + distance(i,i+1) <= drone flight limit (4).
- Dataset is updated with max waiting time and demand of each city.
- NEW: Customer Satisfaction (CS) at each city i is computed as:
         For a normal truck segment: CS(i) = (customer waiting time at i) / (travel time from previous truck city to i),
         For a drone served city: CS(i) = (customer waiting time at i) / (travel time from last truck city to i, using drone speed),
         And for a city served by truck immediately following a drone dispatch (i.e. meeting point), CS is computed as in the normal case,
         while for the subsequent truck city, the effective travel time is: (|drone_time - truck_time| + truck travel time from meeting point to current city).
"""

import numpy as np
def load_coordinates(file_path):
    """
    Đọc dữ liệu tọa độ từ file.
    Mỗi dòng có dạng: [city_id, x, y, max_wait_time]
    Lưu ý: file cần có các giá trị được cách nhau bởi khoảng trắng.
    """
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Giả sử file có: city_id, x, y, max_wait_time
            coordinates.append((float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])))
    return coordinates

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

def calculate_fitness(particle, coordinates, 
                      EV=674.3, ED=14.4,      # Hệ số khí thải của truck và drone
                      alpha=0.01, beta=0.99):  # Trọng số cho khí thải và độ hài lòng khách hàng
    """
    Tính fitness của một giải pháp (particle) dựa trên 2 thành phần:
      1. Khí thải (GHG):
           - Truck: tổng quãng đường truck × EV.
           - Drone: tổng quãng đường drone × ED.
      2. Customer Satisfaction (CS):
           Với mỗi điểm i, CS(i) = (thời gian chờ của khách hàng tại i) / (visited_time tại i),
           trong đó visited_time được xác định dựa trên:
             - Nếu i được phục vụ bởi truck và không có drone xen vào:
                  visited_time = distance(last_truck, current)/50.
             - Nếu i được phục vụ bởi drone:
                  visited_time = distance(last_truck, current)/43.2.
             - Nếu i là điểm truck ngay sau chuỗi drone:
                  * Xác định:
                      x = distance(branch_start, current)/50   (thời gian truck trực tiếp từ branch_start đến current)
                      y = cumulative_time (tổng thời gian của nhánh drone từ branch_start đến current)
                  * waiting_time = max(x, y)
                  * Nếu truck chạy tiếp sau điểm gặp, visited_time được cộng dồn:
                      visited_time_at_current = visited_time_pre + (distance(last_truck, current)/50)
                  * Ban đầu, tại điểm gặp, visited_time_pre = waiting_time.
    
    Fitness = alpha * (total GHG) + beta / (sum CS).
    Mục tiêu: Tối thiểu hóa fitness.
    
    particle: danh sách các tuple (city_id, vehicle) với vehicle = 1 (truck) hoặc 2 (drone).
    coordinates: danh sách các tuple (city_id, x, y, max_wait_time).
    """
    truck_speed = 50
    drone_speed = 43.2

    total_cs = 0.0
    truck_distance = 0.0
    drone_distance = 0.0

    # Khởi tạo: điểm đầu tiên (depot) phục vụ bởi truck
    last_truck = coordinates[int(particle[0][0])]
    branch_start = last_truck       # điểm truck gốc của nhánh drone (nếu có)
    cumulative_time = 0.0           # tích lũy thời gian di chuyển của nhánh drone (theo drone_speed)
    visited_time_pre = 0.0          # thời gian hiệu chỉnh đã tích lũy dùng cho CS của truck sau nhánh drone
    drone_branch_occurred = False   # flag cho biết đã xảy ra nhánh drone hay chưa

    for i in range(1, len(particle)):
        current_city = coordinates[int(particle[i][0])]
        if particle[i][1] == 1:  # Nếu điểm i phục vụ bởi truck
            if particle[i-1][1] == 1:
                # Trường hợp 1: Truck chạy bình thường
                seg_time = euclidean_distance(last_truck, current_city) / truck_speed
                # Nếu chưa có nhánh drone, visited_time_pre chỉ là seg_time;
                # Nếu đã có nhánh drone, ta cần cộng dồn khoảng thời gian từ điểm gặp trước đó.
                if not drone_branch_occurred:
                    visited_time_pre = seg_time
                else:
                    visited_time_pre += seg_time
            else:
                # Trường hợp 3: Truck phục vụ ngay sau một chuỗi drone
                # Tính x: thời gian truck trực tiếp từ branch_start đến current (tính theo truck_speed)
                x = euclidean_distance(branch_start, current_city) / truck_speed
                # y: tổng thời gian của nhánh drone (tích lũy theo drone_speed)
                waiting_time = max(x, cumulative_time)
                visited_time_pre = waiting_time
                drone_branch_occurred = True
            cs = current_city[3] / visited_time_pre if visited_time_pre > 0 else current_city[3]
            total_cs += cs
            truck_distance += euclidean_distance(last_truck, current_city)
            last_truck = current_city
            branch_start = current_city  # Reset branch_start sau khi truck phục vụ
            cumulative_time = 0.0         # Reset cumulative_time
        else:
            # Trường hợp 2: Điểm i phục vụ bởi drone
            seg_time = euclidean_distance(last_truck, current_city) / drone_speed
            if i == 1 or particle[i-1][1] == 1:
                cumulative_time = seg_time
                branch_start = last_truck  # Thiết lập branch_start cho nhánh drone
            else:
                cumulative_time += seg_time
            cs = current_city[3] / seg_time if seg_time > 0 else current_city[3]
            total_cs += cs
            drone_distance += euclidean_distance(last_truck, current_city)
            # Không cập nhật last_truck vì truck vẫn giữ điểm tham chiếu

    ghg_truck = truck_distance * EV
    ghg_drone = drone_distance * ED
    total_ghg = ghg_truck + ghg_drone
    fitness_value = alpha * total_ghg + beta / total_cs
    return fitness_value, truck_distance


# --------------------------
# PHẦN TÌM KIẾM CỤC BỘ DỰA TRÊN FITNESS (VỚI THUẬT TOÁN GREEDY + LOCAL SEARCH)
# --------------------------

def generate_initial_particle(n_cities):
    """
    Sinh giải pháp ban đầu: một permutation ngẫu nhiên của các thành phố (ngoại trừ thành phố 0),
    sau đó chèn thành phố 0 vào đầu và cuối (điểm bắt đầu và kết thúc phải là 0).
    Mỗi thành phố ban đầu được gán xe tải (1).
    """
    cities = list(np.random.permutation(n_cities - 1) + 1)
    particle = [(0, 1)]  # Điểm bắt đầu là 0.
    particle.extend([(city, 1) for city in cities])
    particle.append((0, 1))  # Điểm kết thúc là 0.
    return particle

def dispatch_drone_by_ratio(particle, coordinates, dispatch_ratio=0.3, drone_flight_limit=4):
    """
    Cập nhật lại phân công phương tiện cho lộ trình (particle) theo tỉ lệ.
    Chỉ gán drone (2) nếu thỏa mãn:
      - Thành phố đầu tiên và cuối cùng luôn dùng xe tải (1).
      - Nếu thành phố liền trước đã dùng drone thì buộc thành phố hiện tại phải dùng xe tải.
      - Nếu theo dispatch_ratio có khả năng gán drone, thì kiểm tra:
            * Giả sử A là thành phố trước đó, B là thành phố hiện tại.
            * Tìm kiếm điểm C (trong các thành phố phía sau, trừ điểm cuối) sao cho:
                euclidean_distance(A, B) + euclidean_distance(B, C) <= drone_flight_limit.
            * Nếu tìm được ít nhất 1 điểm C như vậy, gán drone (2) cho B; nếu không, gán xe tải (1).
    """
    n = len(particle)
    updated = []
    # Thành phố đầu tiên luôn dùng xe tải.
    updated.append((particle[0][0], 1))
    
    # Duyệt từ thành phố thứ 1 đến n-2 (bỏ qua điểm cuối)
    for i in range(1, n - 1):
        city_id = particle[i][0]
        if updated[-1][1] == 2:
            updated.append((city_id, 1))
            continue
        
        if np.random.rand() < dispatch_ratio:
            A = coordinates[int(particle[i-1][0])]
            B = coordinates[int(city_id)]
            candidate_found = False
            for k in range(i+1, n - 1):
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
    
    # Thành phố cuối cùng luôn dùng xe tải.
    updated.append((particle[-1][0], 1))
    return updated

def local_search_swap(particle, coordinates):
    """
    Cải thiện giải pháp bằng cách hoán đổi vị trí của hai thành phố (ngoại trừ điểm 0 cố định ở đầu và cuối).
    Trả về giải pháp tốt nhất và fitness tương ứng.
    """
    best_particle = particle[:]
    best_fitness, _ = calculate_fitness(particle, coordinates)
    n = len(particle)
    for i in range(1, n - 1):
        for j in range(i + 1, n - 1):
            new_particle = particle[:]
            new_particle[i], new_particle[j] = new_particle[j], new_particle[i]
            new_particle = dispatch_drone_by_ratio(new_particle, coordinates)
            new_fitness, _ = calculate_fitness(new_particle, coordinates)
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_particle = new_particle[:]
    return best_particle, best_fitness

def local_search_2_opt(particle, coordinates):
    """
    Cải thiện giải pháp bằng thuật toán 2-opt:
    Đảo ngược một đoạn của tour (ngoại trừ điểm đầu và cuối cố định) và cập nhật lại phân công xe.
    Trả về giải pháp tốt nhất và fitness tương ứng.
    """
    best_particle = particle[:]
    best_fitness, _ = calculate_fitness(particle, coordinates)
    n = len(particle)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_particle = particle[:]
            new_particle[i:j+1] = new_particle[i:j+1][::-1]
            new_particle = dispatch_drone_by_ratio(new_particle, coordinates)
            new_fitness, _ = calculate_fitness(new_particle, coordinates)
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_particle = new_particle[:]
    return best_particle, best_fitness

def greedy_search_emission(coordinates, iterations=100, neighborhood_size=4, verbose=True):
    """
    Tìm kiếm giải pháp tối ưu theo tiêu chí fitness (khí thải + CS)
    qua các bước cải thiện:
      - Ban đầu sử dụng local_search_swap.
      - Sau đó, trong mỗi vòng lặp, dùng thuật toán 2-opt để cải thiện tour.
    Lưu lại giải pháp tốt nhất (fitness thấp nhất đạt được).
    """
    n_cities = len(coordinates)
    particle = generate_initial_particle(n_cities)
    particle = dispatch_drone_by_ratio(particle, coordinates)
    best_particle = particle[:]
    best_fitness, truck_distance = calculate_fitness(best_particle, coordinates)
    
    if verbose:
        print("Initial solution:")
        print("Route: " + ", ".join(str(city[0]) for city in best_particle))
        print("Vehicles: " + ", ".join(str(city[1]) for city in best_particle))
        print("Initial fitness: {:.4f}".format(best_fitness))
        print("Initial truck distance: {:.4f}\n".format(truck_distance))
    
    new_particle, new_fitness = local_search_swap(best_particle, coordinates)
    new_fitness, new_truck_distance = calculate_fitness(new_particle, coordinates)
    if verbose:
        print("After greedy swap:")
        print("Route: " + ", ".join(str(city[0]) for city in new_particle))
        print("Vehicles: " + ", ".join(str(city[1]) for city in new_particle))
        print("Fitness: {:.4f}".format(new_fitness))
        print("Truck distance: {:.4f}\n".format(new_truck_distance))
    
    if new_fitness < best_fitness:
        best_particle = new_particle[:]
        best_fitness = new_fitness
        best_fitness, truck_distance = calculate_fitness(best_particle, coordinates)
    
    tol = 1e-6
    for it in range(iterations):
        new_particle, new_fitness = local_search_2_opt(best_particle, coordinates)
        method_used = "2-opt"
        prev_best = best_fitness
        if new_fitness < prev_best - tol:
            best_particle = new_particle[:]
            best_fitness = new_fitness
            best_fitness, truck_distance = calculate_fitness(best_particle, coordinates)
            print(f"Iteration {it+1:3d}: Using {method_used:>8s}, Fitness = {best_fitness:.4f}, Truck distance = {truck_distance:.4f}")
        else:
            print(f"Iteration {it+1:3d}: Using {method_used:>8s}, Fitness = {prev_best:.4f}, Truck distance = {truck_distance:.4f}")
        
    return best_particle, best_fitness, truck_distance

def main():
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\LNS_for_TSPD\\new data\\50.txt"
    coordinates = load_coordinates(file_path)
    
    best_particle, best_fitness, truck_distance = greedy_search_emission(coordinates, iterations=80, neighborhood_size=4, verbose=True)
    best_route = [city[0] for city in best_particle]
    vehicles = [city[1] for city in best_particle]
    
    print("\nFinal best solution (lowest fitness):")
    print("Best route: " + ", ".join(str(x) for x in best_route))
    print("Vehicles (1: Truck, 2: Drone): " + ", ".join(str(x) for x in vehicles))
    print("Best fitness (emission-based): {:.4f}".format(best_fitness))
    print("Total truck distance: {:.4f}".format(truck_distance))

if __name__ == "__main__":
    main()