import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
from shapely.geometry import LineString
from heapq import *
from collections import defaultdict
import math
from math import ceil

# Tính khoảng cách theo đuòng chim bay (Theo công thức Haversine)
def calculate_distance(pa, pb):
    lat1, lon1 = pa
    lat2, lon2 = pb
    
    R = 6371000

    # Chuyển sang radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Công thức Haversine
    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2)**2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def calc_route_distance(G, route, modifier=None):
    total = 0
    for u, v in zip(route[:-1], route[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue

        edge_data = next(iter(data.values()))

        if "length" in edge_data:
            length = edge_data["length"]
            if modifier:
                length *= modifier[(min(u, v), max(u, v))]
            total += length

    return total

def draw_route(G, route, m, color):
    for u, v in zip(route[:-1], route[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue

        edge_data = next(iter(data.values()))

        # Lấy geometry hoặc tự sinh đường thẳng
        if "geometry" in edge_data:
            geom = edge_data["geometry"]
        else:
            point_u = (G.nodes[u]["x"], G.nodes[u]["y"])
            point_v = (G.nodes[v]["x"], G.nodes[v]["y"])
            geom = LineString([point_u, point_v])

        # Folium dùng (lat, lon) nhưng geometry là (lon, lat)
        coords = [(lat, lon) for lon, lat in geom.coords]

        folium.PolyLine(coords, color=color, weight=3).add_to(m)


def render_traffic_route(G, route, traffic_level, m):
    if not route or len(route) < 2:
        return

    # Khởi tạo session_state
    st.session_state.setdefault('edge_modifier', {})
    st.session_state.setdefault('traffic_cache', {})

    # Traffic config
    traffic_scale = [0, 1, 1.5, 2, 2.5, 3, 4, -1]
    if 0 <= traffic_level < len(traffic_scale):
        coeff = traffic_scale[traffic_level]
        display_color = get_traffic_color(traffic_level)
    else:
        coeff = 1
        display_color = 'gray'

    all_coords = []

    for u, v in zip(route[:-1], route[1:]):
        edge_key = (min(u, v), max(u, v))
        st.session_state['edge_modifier'][edge_key] = coeff

        data = G.get_edge_data(u, v)
        coords_segment = []

        if data:
            # Lấy edge đầu tiên có geometry
            for edge_data in data.values():
                geom = edge_data.get('geometry')
                if geom:
                    coords_segment = [(lat, lon) for lon, lat in geom.coords]
                    break
            if not coords_segment:
                # Edge tồn tại nhưng không có geometry
                point_u, point_v = G.nodes[u], G.nodes[v]
                coords_segment = [(point_u['y'], point_u['x']), (point_v['y'], point_v['x'])]
        else:
            # Edge không tồn tại → tìm đường ngắn nhất
            try:
                sub_route = ox.shortest_path(G, u, v, weight='length')
                sub_coords = []
                for uu, vv in zip(sub_route[:-1], sub_route[1:]):
                    sub_data = G.get_edge_data(uu, vv)
                    if sub_data:
                        for sd in sub_data.values():
                            geom = sd.get('geometry')
                            if geom:
                                segment = [(lat, lon) for lon, lat in geom.coords]
                                break
                        else:
                            p1, p2 = G.nodes[uu], G.nodes[vv]
                            segment = [(p1['y'], p1['x']), (p2['y'], p2['x'])]
                    else:
                        p1, p2 = G.nodes[uu], G.nodes[vv]
                        segment = [(p1['y'], p1['x']), (p2['y'], p2['x'])]
                    if sub_coords:
                        sub_coords.extend(segment[1:])
                    else:
                        sub_coords.extend(segment)
                coords_segment = sub_coords
            except Exception as e:
                # Nếu routing thất bại, fallback thẳng
                point_u, point_v = G.nodes[u], G.nodes[v]
                coords_segment = [(point_u['y'], point_u['x']), (point_v['y'], point_v['x'])]

        # Gom coords vào all_coords, bỏ trùng điểm nối
        if all_coords:
            all_coords.extend(coords_segment[1:])
        else:
            all_coords.extend(coords_segment)

        # Lưu cache
        st.session_state['traffic_cache'][edge_key] = traffic_level

    # Vẽ PolyLine liền mạch
    if all_coords:
        folium.PolyLine(
            all_coords,
            color=display_color,
            weight=5,
            opacity=0.8,
            tooltip=f"Mức tắc đường: {traffic_level}"
        ).add_to(m)


def format_distance(meters_value):
    # 1. Làm tròn tổng số mét về số nguyên gần nhất
    total_meters_rounded = round(meters_value)
    
    # 2. Nếu dưới 1000m, trả về mét
    if total_meters_rounded < 1000:
        return f"{total_meters_rounded} m"
    
    # 3. Nếu từ 1000m trở lên
    else:
        kilometers, remaining_meters = divmod(total_meters_rounded, 1000)
        
        # 4. Kiểm tra nếu số mét còn lại là 0
        if remaining_meters == 0:
            return f"{kilometers} km"
        else:
            return f"{kilometers} km {remaining_meters} m"
        
def format_duration(seconds_value):
    # 1. Làm tròn tổng số giây về số nguyên gần nhất
    total_seconds = round(seconds_value)
    
    # 2. Xử lý trường hợp đặc biệt (đầu vào là 0)
    if total_seconds == 0:
        return "0 giây"
    
    # 3. Tính toán các đơn vị
    # Dùng divmod(a, b) trả về (thương, số dư)
    
    # Tính số giờ và số giây còn lại (sau khi trừ đi giờ)
    hours, remaining_seconds_after_hours = divmod(total_seconds, 3600)
    
    # Từ số giây còn lại, tính số phút và số giây cuối cùng
    minutes, seconds = divmod(remaining_seconds_after_hours, 60)
    
    # 4. Xây dựng chuỗi kết quả
    parts = []
    
    if hours > 0:
        parts.append(f"{hours} giờ")
        
    if minutes > 0:
        parts.append(f"{minutes} phút")
        
    if seconds > 0:
        parts.append(f"{seconds} giây")
        
    # 5. Ghép các phần lại với nhau
    # Ví dụ: ["1 giờ", "5 phút", "30 giây"] -> "1 giờ 5 phút 30 giây"
    return " ".join(parts)

def get_traffic_color(level):
    colors = [
        (1, (0, 127, 255)),    # Xanh dương
        (3, (191, 127, 255)),  # Tím nhạt
        (5, (255, 127, 191)),  # Hồng
        (7, (255, 0, 51))      # Đỏ đậm
    ]

    if level <= 1:
        r, g, b = colors[0][1]
        return f'#{r:02x}{g:02x}{b:02x}'
    if level >= 7:
        r, g, b = colors[-1][1]
        return f'#{r:02x}{g:02x}{b:02x}'

    for i in range(len(colors) - 1):
        level_start, (r_start, g_start, b_start) = colors[i]
        level_end, (r_end, g_end, b_end) = colors[i + 1]
        if level_start <= level <= level_end:
            ratio = (level - level_start) / (level_end - level_start)
            r = int(r_start + (r_end - r_start) * ratio)
            g = int(g_start + (g_end - g_start) * ratio)
            b = int(b_start + (b_end - b_start) * ratio)
            return f'#{r:02x}{g:02x}{b:02x}'
    
    return '#007fff'


@st.cache_data
def geocode_address(address):
    if not address:
        # Xử lý trường hợp chuỗi rỗng hoặc None
        st.error("Địa chỉ không được để trống.")
        return None, None
        
    try:
        # ox.geocode trả về một tuple (latitude, longitude)
        lat, lon = ox.geocode(address)
        
        # Log ra console để tiện theo dõi
        print(f"Geocoded '{address}': ({lat}, {lon})") 
        
        # Trả về dưới dạng (lat, lon)
        return lat, lon
        
    except Exception as e:
        # Xử lý lỗi cụ thể (ví dụ: địa chỉ không tìm thấy hoặc lỗi mạng)
        st.error(f"Lỗi Geocoding cho địa chỉ '{address}'. Vui lòng kiểm tra lại địa chỉ. Chi tiết lỗi: {e}")
        return None, None

# Configure OSMnx
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 300
CENTER_LAT = 21.0285
CENTER_LON = 105.8542
ZOOM_START = 16
DEFAULT_LOCATION = (CENTER_LAT, CENTER_LON)  # location Hồ Gươm/trung tâm Hoàn Kiếm
DEFAULT_ZOOM = ZOOM_START
# Define the path to the saved graph file
GRAPHML_FILE = "map.graphml"

# Function to load or create the graph
@st.cache_resource
def load_graph():
    G = ox.load_graphml(GRAPHML_FILE)
    print("Graph loaded successfully.")
    graph = defaultdict(list)
    for edge in G.edges(data=True):
        graph[edge[0]].append((edge[1], edge[2]['length']))
        graph[edge[1]].append((edge[0], edge[2]['length']))
    nodes = dict()
    for node in G.nodes(data=True):
        nodes[node[0]] = (node[1]['y'], node[1]['x'])
    return G, graph, nodes

G, graph, nodes = load_graph()

# Khởi tạo session state
if 'points' not in st.session_state:
    st.session_state['points'] = []
if 'zoom' not in st.session_state:
    st.session_state['zoom'] = DEFAULT_ZOOM
if 'center' not in st.session_state:
    st.session_state['center'] = DEFAULT_LOCATION
if 'traffic_cache' not in st.session_state:
    st.session_state['traffic_cache'] = {}  # Khởi tạo traffic_cache
if 'edit_traffic_mode' not in st.session_state:
    st.session_state['edit_traffic_mode'] = False
if 'traffic_points' not in st.session_state:
    st.session_state['traffic_points'] = []
if 'temp_traffic_level' not in st.session_state:
    st.session_state['temp_traffic_level'] = 1
if 'traffic_click_mode' not in st.session_state:
    st.session_state['traffic_click_mode'] = False  # Trạng thái cho chế độ nhấp chuột
if 'edge_modifier' not in st.session_state:
    st.session_state['edge_modifier'] = {}  # {(u,v): coeff}

m = folium.Map(
    location=st.session_state['center'],
    zoom_start=st.session_state['zoom'],
    tiles='OpenStreetMap',
)

col_start, col_end, col_button = st.columns([1.5, 1.5, 0.5])

# 1. Điểm xuất phát (Trong cột 1)
with col_start:
    start_address = st.text_input("Điểm xuất phát", key="input_start_header") 
    
# 2. Điểm đến (Trong cột 2)
with col_end:
    end_address = st.text_input("Điểm đến", key="input_end_header")

# 3. Nút Tìm đường (Trong cột 3)
with col_button:
    st.markdown("<br>", unsafe_allow_html=True) 
    
    if st.button("Tìm", use_container_width=True):
        if start_address and end_address:
            start_point = geocode_address(start_address)
            end_point = geocode_address(end_address)
            
            # Kiểm tra tọa độ hợp lệ
            if start_point[0] is not None and end_point[0] is not None:
                st.session_state['points'] = [start_point, end_point]
                st.session_state['center'] = start_point
                st.session_state['zoom'] = 16
                st.rerun()
            else:
                # Hiển thị lỗi ở Main Area
                st.error("❌ Không thể tìm thấy tọa độ cho một trong các địa chỉ.") 
        else:
            st.warning("⚠️ Vui lòng nhập cả điểm xuất phát và điểm đến.")
st.text("(Hoặc click chọn trên bản đồ)") 

# Lựa chọn phương tiện
vehicle_type = st.sidebar.selectbox("Chọn phương tiện:",("Đi bộ", "Xe máy", "Ô tô"))

# Thiết lập tốc độ theo phương tiện
if vehicle_type == "Đi bộ":
    speed_mps = 5 * 1000 / 3600   # ~1.3889 m/s
elif vehicle_type == "Xe máy":
    speed_mps = 40 * 1000 / 3600  # ~11.1111 m/s
else:  # ô tô hoặc khác
    speed_mps = 60 * 1000 / 3600 
if 'edge_modifier' not in st.session_state:
    st.session_state['edge_modifier'] = defaultdict(lambda: 1)
def Astar_algorithm(pointA, pointB, time_based=False, editing=False):
    global graph, nodes

    # Priority queue: (f_cost, node)
    queue = []
    heappush(queue, (0, pointA))

    # parent pointer for path reconstruction
    father = {pointA: None}

    # g-cost
    g_cost = {pointA: 0}

    while queue:
        current_f, current = heappop(queue)

        # Found destination
        if current == pointB:
            break

        for neighbor, base_cost in graph[current]:

            key = (min(current, neighbor), max(current, neighbor))

            # Edge penalty/weight
            coeff = st.session_state['edge_modifier'].get(key, 1)

            if editing:
                coeff = 1

            if coeff < 0:
                continue

            if not time_based:
                coeff = 1

            # g(n)
            new_g = g_cost[current] + base_cost * coeff

            # h(n)
            h = calculate_distance(nodes[neighbor], nodes[pointB])
            if time_based:
                h = h / speed_mps

            # f = g + h
            f = new_g + h

            # A* update condition: ONLY check g-cost
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                father[neighbor] = current
                heappush(queue, (f, neighbor))

    # Final distance
    distance = g_cost.get(pointB, float('inf'))

    # Reconstruct path
    path = []
    cur = pointB
    while cur is not None:
        path.append(cur)
        cur = father.get(cur)

    path.reverse()
    return distance, path



for idx, point in enumerate(st.session_state['points']):
    folium.Marker(location=point, tooltip=f"Point {idx+1}", icon=folium.Icon("blue")).add_to(m)

if len(st.session_state['points']) == 2:
    orig, dest = st.session_state['points']
    orig_node = ox.nearest_nodes(G, orig[1], orig[0])
    dest_node = ox.nearest_nodes(G, dest[1], dest[0])

    # 1. Tìm quãng đường ngắn nhất
    distance, route = Astar_algorithm(orig_node, dest_node)

    # 2. Tìm thời gian ngắn nhất
    dis2, route2 = Astar_algorithm(dest_node, orig_node, time_based=True)

    # 3. Tính khoảng cách
    dis_shortest = calc_route_distance(G, route, st.session_state['edge_modifier'])
    dis_fastest = calc_route_distance(G, route2)

    # 4. Tính thời gian
    time_shortest = ceil(dis_shortest / speed_mps)
    time_fastest = ceil(dis_fastest / speed_mps)

    # 5. Vẽ đường đi
    draw_route(G, route, m, color="blue")
    draw_route(G, route2, m, color="green")

    # 6. UI (Show kết quả)
    st.sidebar.markdown("""
    ### Kết quả tìm đường
    - **Phương tiện**: `{}`  
    - **Tốc độ**: `{:.1f}` km/h  
    """.format(vehicle_type, speed_mps * 3.6))

    st.sidebar.markdown(f"1. **Quãng đường ngắn nhất**: `{format_distance(distance)}` trong `{format_duration(time_shortest)}`")

    if dis_fastest > 0:
        st.sidebar.markdown(f"2. **Quãng đường nhanh nhất**: `{format_distance(dis_fastest)}` trong `{format_duration(time_fastest)}`")
    

# Chức năng chỉnh độ tắc đường
# Slider
traffic_level = st.sidebar.slider(
    "Mức độ tắc đường",
    min_value=1,
    max_value=7,
    value=st.session_state.get('temp_traffic_level', 1),
    step=1,
    help="1: xanh – 3: vàng – 5: cam – 7: đỏ (cấm đường)"
)
st.session_state['temp_traffic_level'] = traffic_level

traffic_points = st.session_state.get('traffic_points', [])
if len(traffic_points) == 2:
    orig = traffic_points[0]
    dest = traffic_points[1]

    folium.Marker(orig, tooltip="Điểm đầu", icon=folium.Icon("blue")).add_to(m)
    folium.Marker(dest, tooltip="Điểm cuối", icon=folium.Icon("blue")).add_to(m)

    orig_node = ox.nearest_nodes(G, orig[1], orig[0])
    dest_node = ox.nearest_nodes(G, dest[1], dest[0])

    # A* mode: editing=True → không áp dụng traffic modifier
    _, route = Astar_algorithm(orig_node, dest_node, editing=True)

    if not route:
        st.toast("⚠️ Không tìm thấy đường giữa 2 điểm.")
    else:
        render_traffic_route(G, route, traffic_level, m)

output = st_folium(m, width=1200, height=500, returned_objects=['last_clicked', 'zoom', 'center'])

# nhấp chuột
if output and output['last_clicked']:
    clicked_point = (
        output['last_clicked']['lat'],
        output['last_clicked']['lng']
    )

    st.session_state['zoom'] = output['zoom']
    st.session_state['center'] = (
        output['center']['lat'],
        output['center']['lng']
    )

    #  CLICK TRAFFIC MODE
    if st.session_state['edit_traffic_mode'] and st.session_state['traffic_click_mode']:

        if clicked_point not in st.session_state['traffic_points'] \
            and len(st.session_state['traffic_points']) < 2:
            st.session_state['traffic_points'].append(clicked_point)

        # rerun NẾU đủ 2 điểm
        if len(st.session_state['traffic_points']) == 2:
            st.rerun()

    #  CLICK TÌM ĐƯỜNG
    if not st.session_state['edit_traffic_mode']:
        if len(st.session_state['points']) < 2:
            st.session_state['traffic_points'].append(clicked_point)
            st.session_state['points'].append(clicked_point)

        if len(st.session_state['points']) == 2:
            st.rerun()

# Tắc đường
if st.sidebar.button("Chỉnh sửa tắc đường"):
    st.session_state['edit_traffic_mode'] = True
    st.session_state['traffic_points'] = []
    st.session_state['temp_traffic_level'] = 1
    st.session_state['traffic_click_mode'] = 1
    st.rerun()


if st.sidebar.button(label = "Đặt lại tất cả", type="secondary"):
    st.session_state['traffic_cache'] = {}
    st.session_state['edge_modifier'] = defaultdict(lambda: 1)
    st.session_state['edit_traffic_mode'] = False
    st.session_state['traffic_points'] = []
    st.session_state['temp_traffic_level'] = 1
    st.session_state['traffic_click_mode'] = False
    st.session_state['points'] = []
    st.session_state['zoom'] = DEFAULT_ZOOM
    st.session_state['center'] = DEFAULT_LOCATION
    st.rerun()