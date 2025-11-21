import osmnx as ox

# ĐỊNH NGHĨA KHU VỰC CẦN TẢI
place_name = "Hoan Kiem, Hanoi, Vietnam"

try:
    # 1. Tải Graph
    G = ox.graph_from_place(place_name, network_type="drive")
    print(f"Đã tải thành công mạng lưới đường bộ cho: {place_name}")

    # 2. Xử lý Dữ liệu và Tối ưu
    # Hàm add_edge_speeds hiện tại đã bao gồm việc tính travel_time
    G = ox.add_edge_speeds(G) 
    
    # BỎ QUA dòng bị lỗi G = ox.add_travel_times(G)

    # 3. LƯU GRAPH VÀO FILE
    output_filename = 'hanoi_new.graphml'
    ox.save_graphml(G, filepath=output_filename)
    print(f"Đã lưu Graph vào file: {output_filename}")

except Exception as e:
    print(f"Lỗi khi tải Graph: {e}")