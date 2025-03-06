import copy
import numpy as np
import random

############################################################################
# Hàm: load_data
############################################################################
def load_data():
    file_path = "D:\\Tran Hoang Vu\\Lab\\VRDP\\Large Nearest Search (LNS)\\lns\\data\\48.txt"
    # Đọc dữ liệu không ép kiểu để giữ giá trị thực
    data = np.loadtxt(file_path)
    N = len(data)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                # Tính khoảng cách Euclidean giữa điểm i và điểm j
                dist_matrix[i][j] = np.sqrt((data[i, 1] - data[j, 1]) ** 2 + (data[i, 2] - data[j, 2]) ** 2)
    return dist_matrix

############################################################################
# Hàm: Euclidean Distance (không bắt buộc dùng)
############################################################################
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

############################################################################
# Hàm: distance_calc
# Tính tổng khoảng cách cho một tour dưới dạng danh sách gồm 2 phần tử:
# phần tử thứ nhất là tour (danh sách các thành phố theo thứ tự, dùng chỉ số 1-index)
# phần tử thứ hai là khoảng cách hiện tại.
############################################################################
def distance_calc(distance_matrix, city_tour):
    distance = 0
    # Lưu ý: city_tour[0] chứa tour với các thành phố được đánh số từ 1 trở đi.
    for k in range(0, len(city_tour[0]) - 1):
        m = k + 1
        distance += distance_matrix[city_tour[0][k] - 1, city_tour[0][m] - 1]
    return distance

############################################################################
# Hàm: distance_point
# Tính tổng khoảng cách cho tour dưới dạng danh sách các chỉ số (0-index).
############################################################################
def distance_point(distance_matrix, city_tour):
    distance = 0
    for i in range(len(city_tour) - 1):
        distance += distance_matrix[city_tour[i]][city_tour[i + 1]]
    distance += distance_matrix[city_tour[-1]][city_tour[0]]
    return distance

############################################################################
# Hàm: local_search_2_opt
# Thực hiện cải tiến tour theo phương pháp 2-opt.
############################################################################
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding=-1, verbose=True):
    if recursive_seeding < 0:
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance = city_list[1] * 2
    iteration = 0
    if verbose:
        print("\nLocal Search\n")
    while count < recursive_seeding:
        if verbose:
            print("Iteration =", iteration, "Distance =", round(city_list[1], 2))
        best_route = copy.deepcopy(city_list)
        seed = copy.deepcopy(city_list)
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i + 1, len(city_list[0]) - 1):
                # Thực hiện đảo ngược đoạn tour từ i đến j
                best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))
                # Đảm bảo tour khép kín
                best_route[0][-1] = best_route[0][0]
                best_route[1] = distance_calc(distance_matrix, best_route)
                if city_list[1] > best_route[1]:
                    city_list = copy.deepcopy(best_route)
                best_route = copy.deepcopy(seed)
        count = count + 1
        iteration = iteration + 1
        if distance > city_list[1] and recursive_seeding < 0:
            distance = city_list[1]
            count = -2
            recursive_seeding = -1
        elif city_list[1] >= distance and recursive_seeding < 0:
            count = -1
            recursive_seeding = -2
    return city_list[0], city_list[1]

############################################################################
# Hàm: removal_operators
# Định nghĩa các hàm loại bỏ điểm khỏi tour.
############################################################################
def removal_operators():
    def random_removal(city_tour, num_removals):
        # Lưu ý: Với tour dạng list các thành phố (0-index), bỏ qua thành phố đầu tiên nếu cần
        removed = set()
        while len(removed) < num_removals:
            removed.add(random.choice(city_tour[1:]))
        return list(removed)
    return [random_removal]

############################################################################
# Hàm: insertion_operators
# Định nghĩa các hàm chèn điểm vào tour.
############################################################################
def insertion_operators():
    def cheapest_insertion(removed_nodes, city_tour, distance_matrix):
        for node in removed_nodes:
            best_insertion_cost = float('inf')
            best_insertion_index = -1
            for i in range(1, len(city_tour) + 1):
                insertion_cost = (distance_matrix[city_tour[i - 1]][node] +
                                  distance_matrix[node][city_tour[i % len(city_tour)]] -
                                  distance_matrix[city_tour[i - 1]][city_tour[i % len(city_tour)]])
                if insertion_cost < best_insertion_cost:
                    best_insertion_cost = insertion_cost
                    best_insertion_index = i
            city_tour.insert(best_insertion_index, node)
        return city_tour
    return [cheapest_insertion]

############################################################################
# Hàm: adaptive_large_neighborhood_search (ALNS)
# Thuật toán tìm kiếm lân cận lớn thích ứng.
############################################################################
def adaptive_large_neighborhood_search(distance_matrix, iterations=100, removal_fraction=0.2, rho=0.1, local_search=True, verbose=True):
    # Khởi tạo tour ban đầu (0-index)
    initial_tour = list(range(distance_matrix.shape[0]))
    random.shuffle(initial_tour)
    route = initial_tour.copy()
    distance = distance_point(distance_matrix, route)
    
    removal_ops = removal_operators()
    insertion_ops = insertion_operators()
    weights_removal = [1.0] * len(removal_ops)
    weights_insertion = [1.0] * len(insertion_ops)
    count = 0
    
    while count <= iterations:
        if verbose and count > 0:
            print("Iteration =", count, "Distance =", round(distance, 2))
        city_tour = route.copy()
        removal_op = random.choices(removal_ops, weights=weights_removal)[0]
        insertion_op = random.choices(insertion_ops, weights=weights_insertion)[0]
        num_removals = int(removal_fraction * distance_matrix.shape[0])
        removed_nodes = removal_op(city_tour, num_removals)
        for node in removed_nodes:
            city_tour.remove(node)
        new_tour = insertion_op(removed_nodes, city_tour, distance_matrix)
        new_tour_distance = distance_point(distance_matrix, new_tour)
        if new_tour_distance < distance:
            route = new_tour
            distance = new_tour_distance
            weights_removal[removal_ops.index(removal_op)] *= (1 + rho)
            weights_insertion[insertion_ops.index(insertion_op)] *= (1 + rho)
        else:
            weights_removal[removal_ops.index(removal_op)] *= (1 - rho)
            weights_insertion[insertion_ops.index(insertion_op)] *= (1 - rho)
        total_weight_removal = sum(weights_removal)
        total_weight_insertion = sum(weights_insertion)
        weights_removal = [w / total_weight_removal for w in weights_removal]
        weights_insertion = [w / total_weight_insertion for w in weights_insertion]
        count += 1

    # Đóng tour bằng cách nối điểm đầu tiên vào cuối danh sách
    route = route + [route[0]]
    # Chuyển sang đánh số thành phố dạng 1-index để dễ đọc
    route = [item + 1 for item in route]
    
    if local_search:
        # Gói tour và khoảng cách vào list [tour, distance] để chạy 2-opt
        route, distance = local_search_2_opt(distance_matrix, [route, distance], -1, verbose)
    return route, distance

############################################################################
# Phần Main: Chạy toàn bộ thuật toán
############################################################################
if __name__ == '__main__':
    # Tải dữ liệu từ file và xây dựng ma trận khoảng cách
    dist_matrix = load_data()
    
    # Chạy thuật toán ALNS
    route, distance = adaptive_large_neighborhood_search(
        dist_matrix, 
        iterations=100, 
        removal_fraction=0.2, 
        rho=0.1, 
        local_search=True, 
        verbose=True
    )
    
    print("\nOptimal Route:", route)
    print("Optimal Distance:", distance)
