"""
- This script is updated with emission fitness function and some constraints between truck and drone such as:
    + limit distance of drone = 4
    + have a raito of waiting time between truck and drone
    + drone is dispatched and identify quite exactly a return location with fomular:
        distance(i-1,i) + distance(i,i+1) <= limit drone distance
- Dataset is update with max waiting time and demand of each city
"""

import numpy as np

# --------------------------
# PHẦN ĐỌC FILE VÀ TÍNH FITNESS (DỰA TRÊN KHÍ THẢI)
# --------------------------

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
                      EV=674.3, ED=14.4,      # Hệ số khí thải: truck và drone
                      alpha=0.01, beta=0.99):  # Trọng số cho khí thải và độ hài lòng khách hàng
    """
    Tính fitness của một giải pháp (particle) dựa trên:
      - Tổng khí thải (GHG) từ xe tải và drone:
           + Xe tải: quãng đường × 674.3.
           + Drone: quãng đường × 14.4.
      - Customer satisfaction (CS) được tính qua penalty thời gian chờ:
           + Với xe tải: tại mỗi điểm, CS = (visited_time / max_wait_time), trong đó visited_time là 
             một giá trị ngẫu nhiên từ 0 đến max_wait_time của điểm đó.
           + Với drone: khi drone phóng từ A (particle[i-1]) đến B (particle[i]),
             sẽ tìm một điểm C (với vehicle == 1) phía sau sao cho:
                 - Khoảng cách từ B đến C cộng với khoảng cách từ A đến B không vượt quá drone flight limit (4).
                 - Tính truck_time = distance(A, C)/50 và 
                   drone_time = (distance(A, B) + distance(B, C))/43.2.
                 - Waiting_ratio = |truck_time - drone_time| / max_wait_time của C.
             Penalty của drone sẽ được cộng vào tổng CS theo waiting_ratio nhỏ nhất tìm được.
    
    particle: danh sách các tuple (city_id, vehicle) với city_id là số nguyên 0-indexed.
              (vehicle: 1 cho xe tải, 2 cho drone)
    coordinates: danh sách các tuple (city_id, x, y, max_wait_time)
    
    Trả về: fitness_value, truck_distance (quãng đường xe tải).
    
    Lưu ý: Mục tiêu là tối thiểu hóa hàm fitness.
    """
    total_ghg = 0
    total_cs = 0
    dist_vehicle = 0
    dist_drone = 0

    truck_speed = 50
    drone_speed = 43.2
    drone_flight_limit = 4

    # Biến current_time để lưu lại thời điểm truck đến từng thành phố
    current_time = 0

    # Tính quãng đường di chuyển và CS cho xe tải
    for i in range(len(particle)):
        current_city = coordinates[int(particle[i][0])]
        current_vehicle = particle[i][1]
        if current_vehicle == 1:  # Xe tải
            if i > 0:
                prev_city = coordinates[int(particle[i - 1][0])]
                travel_time = euclidean_distance(prev_city, current_city) / truck_speed
                current_time += travel_time
                dist_vehicle += euclidean_distance(prev_city, current_city)
            # CS cho xe tải: visited_time là thời điểm truck đến hiện tại (current_time)
            max_wait_time = current_city[3]
            # Giả sử rằng thời gian chờ lý tưởng là càng ít càng tốt, nên CS tính theo tỷ lệ:
            cs = current_time / max_wait_time
            total_cs += cs
        else:  # Drone
            if i > 0:
                prev_city = coordinates[int(particle[i - 1][0])]
                dist_drone += euclidean_distance(prev_city, current_city)
    # Tính khí thải theo khoảng cách di chuyển của xe tải và drone
    ghg_vehicle = dist_vehicle * EV
    ghg_drone = dist_drone * ED
    total_ghg = ghg_vehicle + ghg_drone

    # Tính penalty thời gian chờ cho các đoạn drone (vehicle == 2)
    for i in range(1, len(particle)):
        if particle[i][1] == 2:
            A = coordinates[int(particle[i-1][0])]  # Điểm phóng drone
            B = coordinates[int(particle[i][0])]       # Điểm giao hàng của drone
            min_waiting_ratio = float('inf')
            for k in range(i+1, len(particle)):
                if particle[k][1] == 1:  # Chỉ xét các thành phố phục vụ bởi xe tải
                    C = coordinates[int(particle[k][0])]
                    # Điều kiện: tổng khoảng cách từ A đến B và từ B đến C không vượt quá giới hạn bay của drone
                    if euclidean_distance(A, B) + euclidean_distance(B, C) <= drone_flight_limit:
                        truck_time = euclidean_distance(A, C) / truck_speed
                        drone_time = (euclidean_distance(A, B) + euclidean_distance(B, C)) / drone_speed
                        waiting_time = abs(truck_time - drone_time)
                        waiting_ratio = waiting_time / C[3] if C[3] != 0 else waiting_time
                        if waiting_ratio < min_waiting_ratio:
                            min_waiting_ratio = waiting_ratio
            if min_waiting_ratio < float('inf'):
                total_cs += min_waiting_ratio

    fitness_value = alpha * total_ghg + beta * total_cs
    return fitness_value, dist_vehicle

# --------------------------
# PHẦN TÌM KIẾM CỤC BỘ DỰA TRÊN FITNESS (VỚI THUẬT TOÁN GREEDY + LOCAL SEARCH)
# --------------------------

def generate_initial_particle(n_cities):
    """
    Sinh giải pháp ban đầu: một permutation ngẫu nhiên của các thành phố (ngoại trừ thành phố 0),
    sau đó chèn thành phố 0 vào đầu và cuối (điểm bắt đầu và kết thúc phải là 0).
    Mỗi thành phố được gán xe tải (1).
    """
    # Giả sử n_cities là tổng số thành phố, các thành phố được đánh số từ 0 đến n_cities-1.
    cities = list(np.random.permutation(n_cities - 1) + 1)  # Sinh permutation cho các thành phố 1..n_cities-1.
    particle = [(0, 1)]            # Điểm bắt đầu là thành phố 0.
    particle.extend([(city, 1) for city in cities])
    particle.append((0, 1))        # Điểm kết thúc cũng là thành phố 0.
    return particle

def dispatch_drone_by_ratio(particle, coordinates, dispatch_ratio=0.3, drone_flight_limit=4):
    """
    Cập nhật lại phân công phương tiện cho lộ trình (particle) theo tỉ lệ,
    nhưng chỉ gán drone (2) nếu thỏa mãn các điều kiện sau:
      - Thành phố đầu tiên (và cuối cùng) luôn dùng xe tải (1).
      - Nếu thành phố liền trước đã được gán drone thì buộc thành phố hiện tại phải dùng xe tải.
      - Nếu theo dispatch_ratio, có khả năng gán drone, thì thực hiện phóng drone từ A -> B:
            * A là thành phố trước đó (nơi drone được phóng),
            * B là thành phố hiện tại (điểm giao hàng của drone).
        Sau đó, xét điểm B với tọa độ từ coordinates, tìm xem có tồn tại một điểm gặp (C)
        trong các thành phố phía sau sao cho:
                  euclidean_distance(A, B) + euclidean_distance(B, C) <= drone_flight_limit.
        Nếu tìm được ít nhất một điểm như vậy, gán drone (2) cho điểm hiện tại;
        nếu không, gán xe tải (1).
    
    Args:
        particle: Danh sách các tuple (city_id, vehicle) của lộ trình ban đầu.
                  (Lưu ý: thành phố đầu và cuối phải là 0.)
        coordinates: Danh sách các tuple (city_id, x, y, max_wait_time).
        dispatch_ratio: Xác suất mong muốn gán drone cho mỗi thành phố.
        drone_flight_limit: Giới hạn bay của drone.
        
    Returns:
        Một danh sách particle mới với phân công phương tiện được cập nhật.
    """
    n = len(particle)
    updated = []
    # Thành phố đầu tiên luôn dùng xe tải.
    updated.append((particle[0][0], 1))
    
    # Duyệt từ thành phố thứ 1 đến n-2, bỏ qua thành phố cuối cùng vì nó phải là 0
    for i in range(1, n - 1):
        city_id = particle[i][0]
        # Nếu thành phố liền trước đã dùng drone thì buộc thành phố hiện tại phải dùng xe tải.
        if updated[-1][1] == 2:
            updated.append((city_id, 1))
            continue
        
        if np.random.rand() < dispatch_ratio:
            A = coordinates[int(particle[i-1][0])]
            B = coordinates[int(city_id)]
            candidate_found = False
            # Tìm kiếm điểm C trong các thành phố phía sau (trừ thành phố cuối cùng 0)
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
    Cải thiện giải pháp bằng hoán đổi vị trí của hai thành phố (trừ điểm 0 cố định ở đầu và cuối).
    Trả về giải pháp tốt nhất tìm được và fitness tương ứng.
    """
    best_particle = particle[:]
    best_fitness, _ = calculate_fitness(particle, coordinates)
    n = len(particle)
    # Duyệt qua các vị trí từ 1 đến n-2 (không thay đổi điểm đầu và cuối)
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
    Đảo ngược một đoạn của tour (trừ điểm đầu và cuối cố định) và cập nhật lại phân công xe.
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
      - Đầu tiên dùng local_search_swap (đập đi xây lại theo kiểu tham lam)
      - Sau đó, trong mỗi vòng lặp, sử dụng 2-opt để cải thiện tour.
    Nếu giải pháp của vòng sau không cải thiện (fitness không giảm) thì in ra kết quả của vòng trước.
    Lưu lại toàn cục best solution (fitness thấp nhất đạt được).
    (Bỏ qua bước random perturbation)
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
        print("After greedy removal/insertion (simulated by swap):")
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
