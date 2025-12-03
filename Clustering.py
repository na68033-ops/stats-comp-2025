import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import folium
pd.options.display.max_columns = None

# 문화 시설
ulsan_mun = pd.read_csv('C:/Users/data/ulsan/ULSAN_MUN_DATA.csv',encoding='euc-kr')
ulsan_mun.info()

ulsan_mun_f = ulsan_mun[['아이디(ID)', '여행지명', '대구분', '중구분', '소구분', '위도값', '경도값']].copy()
ulsan_mun_f = ulsan_mun_f.drop_duplicates()
ulsan_mun_r = ulsan_mun_f.rename(columns={'아이디(ID)':'ID', '여행지명':'TRANM', '대구분':'SEG1', '중구분':'SEG2', '소구분':'SEG3', '위도값':'LATITUDE', '경도값':'LONGITUDE'})

# 여행지명
ulsan_tra = pd.read_csv('C:/Users/data/ulsan/ULSAN_TRA_DATA.csv',encoding='euc-kr')
ulsan_tra.info()

ulsan_tra_f = ulsan_tra[['아이디(ID)','여행지명', '대구분', '중구분', '소구분', '위도값', '경도값']].copy()
ulsan_tra_f = ulsan_tra_f.drop_duplicates()
ulsan_tra_r = ulsan_tra_f.rename(columns={'아이디(ID)':'ID','여행지명':'TRANM', '대구분':'SEG1', '중구분':'SEG2', '소구분':'SEG3', '위도값':'LATITUDE', '경도값':'LONGITUDE'})

# 문화시설 + 여행지명
ulsan_data = pd.concat([ulsan_mun_r, ulsan_tra_r], axis = 0).reset_index(drop = True)
ulsan_data = ulsan_data.drop_duplicates()

# 함께검색 건수 : 관광지 중요도
ulsan_cnt = pd.read_csv('C:/Users/data/ulsan/ULSAN_TO_COUNT.csv',encoding='cp949')
ulsan_cnt.info()

# 관광지 중요도 merge
ulsan_data1 = pd.merge(ulsan_data, ulsan_cnt, on = 'ID', how = 'left')
ulsan_data1.info()



# 군집분석 Start

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point, LineString
import networkx as nx
from geopy.distance import great_circle
import itertools
from scipy.spatial.distance import pdist
import geopandas as gpd

# =====================================
# 1. 데이터 로드 및 전처리
# =====================================
# ulsan_data1 = pd.read_csv("your_data.csv")  # 실제 데이터 로드
# 필요한 컬럼: LATITUDE, LONGITUDE, COUNT (이용건수)

# =====================================
# 2. DBSCAN 군집분석 (가중치 반영)
# =====================================

# 좌표 변환
coords = ulsan_data1[['LATITUDE', 'LONGITUDE']].values

# eps 계산 (2km 반경)
kms_per_radian = 6371.0088
eps = 2 / kms_per_radian  # 2km 기준

# 이용건수를 0~1로 정규화
ulsan_data1['Value'] = ulsan_data1['COUNT'] / ulsan_data1['COUNT'].max()

# 인기 관광지의 영향력을 좌표 가중치로 반영
weighted_coords = np.column_stack([
    ulsan_data1['LATITUDE'] + (ulsan_data1['Value'] - 0.5) * 0.002,
    ulsan_data1['LONGITUDE'] + (ulsan_data1['Value'] - 0.5) * 0.002
])

# DBSCAN 실행
db = DBSCAN(eps=eps, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(weighted_coords))
cluster_labels = db.labels_

ulsan_data1['cluster'] = cluster_labels

print(f"총 군집 수: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
print(f"노이즈 포인트 수: {sum(cluster_labels == -1)}")

# =====================================
# 3. 군집별 통계 및 품질 평가
# =====================================

clusters = []
for cluster_id in set(cluster_labels):
    if cluster_id == -1:
        continue  # 노이즈 제외

    cluster_data = ulsan_data1[ulsan_data1['cluster'] == cluster_id]
    cluster_points = cluster_data[['LATITUDE', 'LONGITUDE']].values
    centroid = MultiPoint(cluster_points).centroid

    # 군집 응집도 계산 (군집 내 평균 거리)
    compactness = 0.0
    if len(cluster_points) > 1:
        # haversine 직접 계산
        from itertools import combinations

        distances = []
        for (lat1, lon1), (lat2, lon2) in combinations(cluster_points, 2):
            dist = great_circle((lat1, lon1), (lat2, lon2)).km
            distances.append(dist)
        compactness = np.mean(distances) if distances else 0.0

    clusters.append({
        'cluster_id': cluster_id,
        'center_lat': centroid.y,
        'center_lon': centroid.x,
        'tour_spot_count': len(cluster_data),
        'avg_usage': cluster_data['COUNT'].mean(),
        'total_usage': cluster_data['COUNT'].sum(),
        'compactness_km': round(compactness, 2)
    })

cluster_summary = pd.DataFrame(clusters)
cluster_summary = cluster_summary.sort_values('total_usage', ascending=False).reset_index(drop=True)

print(cluster_summary.to_string(index=False))

# =====================================
# 4. 네트워크 그래프 생성 (수요 기반 가중치)
# =====================================
# 위도/경도가 바뀌었는지 확인 (울산은 위도 35°, 경도 129° 부근)
if cluster_summary['center_lat'].mean() > 90 or cluster_summary['center_lat'].mean() < -90:
    print("\n⚠️  WARNING: Latitude/Longitude 컬럼이 바뀐 것으로 보입니다. 자동 교정합니다.")
    cluster_summary.rename(columns={'center_lat': 'center_lon', 'center_lon': 'center_lat'}, inplace=True)
    print(f"교정 후 Latitude 범위: {cluster_summary['center_lat'].min():.6f} ~ {cluster_summary['center_lat'].max():.6f}")
    print(f"교정 후 Longitude 범위: {cluster_summary['center_lon'].min():.6f} ~ {cluster_summary['center_lon'].max():.6f}")

# 군집 간 거리 계산
edges = []
for (i, j) in itertools.combinations(range(len(cluster_summary)), 2):
    point_i = (cluster_summary.iloc[i]['center_lat'], cluster_summary.iloc[i]['center_lon'])
    point_j = (cluster_summary.iloc[j]['center_lat'], cluster_summary.iloc[j]['center_lon'])
    dist = great_circle(point_i, point_j).km
    edges.append((cluster_summary.iloc[i]['cluster_id'],
                  cluster_summary.iloc[j]['cluster_id'],
                  dist))

# 네트워크 그래프 생성
G = nx.Graph()
for _, row in cluster_summary.iterrows():
    G.add_node(
        row['cluster_id'],
        pos=(row['center_lon'], row['center_lat']),
        weight=row['total_usage']
    )

# 수요 기반 가중치로 엣지 추가
for a, b, dist in edges:
    weight_a = cluster_summary[cluster_summary['cluster_id'] == a]['total_usage'].values[0]
    weight_b = cluster_summary[cluster_summary['cluster_id'] == b]['total_usage'].values[0]

    # 가중치: 거리 / (두 군집 수요의 합)
    # → 수요가 높을수록 가중치 감소 = 우선 연결
    adjusted_weight = dist / (weight_a + weight_b + 1)

    G.add_edge(a, b, weight=adjusted_weight, distance=dist)

print(f"노드 수: {G.number_of_nodes()}")
print(f"엣지 수: {G.number_of_edges()}")

# =====================================
# 5. MST 기반 기본 노선 네트워크
# =====================================
print("\n" + "=" * 50)
print("STEP 4: MST 기반 기본 노선 생성")
print("=" * 50)

mst = nx.minimum_spanning_tree(G, weight='weight')

routes_mst = []
for u, v, data in mst.edges(data=True):
    actual_distance = G[u][v]['distance']
    routes_mst.append({
        'start_cluster': u,
        'end_cluster': v,
        'distance_km': round(actual_distance, 2),
        'adjusted_weight': round(data['weight'], 4)
    })

route_mst_df = pd.DataFrame(routes_mst)

# 우선순위 표시 (상위 3개 군집 포함 노선)
top_clusters = cluster_summary.head(3)['cluster_id'].tolist()
route_mst_df['priority'] = route_mst_df.apply(
    lambda x: 1 if (x['start_cluster'] in top_clusters or x['end_cluster'] in top_clusters) else 0,
    axis=1
)

print("\n[MST 기반 노선]")
print(route_mst_df.to_string(index=False))

# =====================================
# 6. TSP 기반 순환 노선 (상위 N개 군집)
# =====================================
print("\n" + "=" * 50)
print("STEP 5: TSP 기반 순환 노선 생성")
print("=" * 50)

# 상위 5개 인기 군집 선택
top_n = min(5, len(cluster_summary))
top_n_clusters = cluster_summary.head(top_n)['cluster_id'].tolist()

print(f"순환 노선 대상 군집: {top_n_clusters}")

# 부분 그래프 생성
subgraph = G.subgraph(top_n_clusters)

try:
    # Greedy TSP 근사
    tsp_path = nx.approximation.greedy_tsp(subgraph, weight='weight', source=top_n_clusters[0])

    # 경로를 순서대로 저장
    tsp_routes = []
    total_distance = 0

    for i in range(len(tsp_path) - 1):
        u, v = tsp_path[i], tsp_path[i + 1]
        dist = G[u][v]['distance']
        total_distance += dist

        tsp_routes.append({
            'sequence': i + 1,
            'from_cluster': u,
            'to_cluster': v,
            'distance_km': round(dist, 2)
        })

    tsp_route_df = pd.DataFrame(tsp_routes)

    print("\n[TSP 순환 노선]")
    print(tsp_route_df.to_string(index=False))
    print(f"\n총 순환 거리: {round(total_distance, 2)} km")

except Exception as e:
    print(f"TSP 경로 생성 실패: {e}")
    print("MST 결과를 사용하세요.")
    tsp_route_df = None

# =====================================
# 7. CSV 결과 저장
# =====================================

output_path = "C:/Users/data/ulsan/"

# 원본 데이터 (군집 정보 포함)
ulsan_data1.to_csv(f"{output_path}ulsan_tour_clusters.csv", index=False, encoding='utf-8-sig')
print(f"✓ {output_path}ulsan_tour_clusters.csv")

# 군집 요약
cluster_summary.to_csv(f"{output_path}ulsan_cluster_summary.csv", index=False, encoding='utf-8-sig')
print(f"✓ {output_path}ulsan_cluster_summary.csv")

# MST 노선
route_mst_df.to_csv(f"{output_path}ulsan_yegaro_mst_routes.csv", index=False, encoding='utf-8-sig')
print(f"✓ {output_path}ulsan_yegaro_mst_routes.csv")

# TSP 순환 노선
if tsp_route_df is not None:
    tsp_route_df.to_csv(f"{output_path}ulsan_yegaro_tsp_route.csv", index=False, encoding='utf-8-sig')
    print(f"✓ {output_path}ulsan_yegaro_tsp_route.csv")

# =====================================
# 8. QGIS 시각화용 GeoJSON 생성
# =====================================
print("\n" + "=" * 50)
print("STEP 7: GeoJSON 생성 (QGIS용)")
print("=" * 50)

# 군집 중심점 GeoDataFrame
gdf_clusters = gpd.GeoDataFrame(
    cluster_summary,
    geometry=[Point(row['center_lon'], row['center_lat']) for _, row in cluster_summary.iterrows()],
    crs='EPSG:4326'
)

# MST 노선 LineString
mst_geometries = []
for _, route in route_mst_df.iterrows():
    start = cluster_summary[cluster_summary['cluster_id'] == route['start_cluster']].iloc[0]
    end = cluster_summary[cluster_summary['cluster_id'] == route['end_cluster']].iloc[0]
    line = LineString([
        (start['center_lon'], start['center_lat']),
        (end['center_lon'], end['center_lat'])
    ])
    mst_geometries.append(line)

gdf_mst_routes = gpd.GeoDataFrame(route_mst_df, geometry=mst_geometries, crs='EPSG:4326')

# TSP 노선 LineString
if tsp_route_df is not None:
    tsp_geometries = []
    for _, route in tsp_route_df.iterrows():
        start = cluster_summary[cluster_summary['cluster_id'] == route['from_cluster']].iloc[0]
        end = cluster_summary[cluster_summary['cluster_id'] == route['to_cluster']].iloc[0]
        line = LineString([
            (start['center_lon'], start['center_lat']),
            (end['center_lon'], end['center_lat'])
        ])
        tsp_geometries.append(line)

    gdf_tsp_routes = gpd.GeoDataFrame(tsp_route_df, geometry=tsp_geometries, crs='EPSG:4326')

# GeoJSON 저장
gdf_clusters.to_file(f"{output_path}ulsan_clusters.geojson", driver='GeoJSON', encoding='utf-8')
print(f"✓ {output_path}ulsan_clusters.geojson")

gdf_mst_routes.to_file(f"{output_path}ulsan_mst_routes.geojson", driver='GeoJSON', encoding='utf-8')
print(f"✓ {output_path}ulsan_mst_routes.geojson")

if tsp_route_df is not None:
    gdf_tsp_routes.to_file(f"{output_path}ulsan_tsp_route.geojson", driver='GeoJSON', encoding='utf-8')
    print(f"✓ {output_path}ulsan_tsp_route.geojson")

# 개별 관광지 포인트 (군집 정보 포함)
gdf_points = gpd.GeoDataFrame(
    ulsan_data1,
    geometry=[Point(row['LONGITUDE'], row['LATITUDE']) for _, row in ulsan_data1.iterrows()],
    crs='EPSG:4326'
)
gdf_points.to_file(f"{output_path}ulsan_tour_spots.geojson", driver='GeoJSON', encoding='utf-8')
print(f"✓ {output_path}ulsan_tour_spots.geojson")

print("\n" + "=" * 50)
print("분석 완료!")
print("=" * 50)
print("\n📊 QGIS 시각화 가이드:")
print("1. ulsan_clusters.geojson - 군집 중심점 (크기: total_usage)")
print("2. ulsan_mst_routes.geojson - MST 기반 전체 네트워크")
print("3. ulsan_tsp_route.geojson - 상위 군집 순환 노선")
print("4. ulsan_tour_spots.geojson - 개별 관광지 (색상: cluster)")






# =====================================
# 클러스터 2 특정 관광지만 필터링
# =====================================

# 필터링할 관광지 목록
target_spots = [
    '간절곶',
    '명선교',
    '울산옹기박물관',
    '울산해양박물관',
    '울주민속박물관',
    '외고산 옹기마을',
    '진하해수욕장'
]

# 데이터 로드 (이미 군집분석이 완료된 파일)
# ulsan_data1이 메모리에 있다면 바로 사용, 없다면 CSV에서 로드
try:
    df = ulsan_data1.copy()
except NameError:
    # CSV 파일에서 로드
    df = pd.read_csv("C:/Users/data/ulsan/ulsan_tour_clusters.csv", encoding='utf-8-sig')

# 관광지명 컬럼 확인 (실제 컬럼명에 맞게 수정 필요)
# 가능한 컬럼명: 'NAME', 'TOUR_NAME', 'SPOT_NAME' 등
# 아래 코드에서 'NAME'을 실제 컬럼명으로 변경하세요
name_column = 'TRANM_x'  # ← 실제 관광지명 컬럼으로 변경

# 컬럼 존재 확인
if name_column not in df.columns:
    print(f"⚠️  '{name_column}' 컬럼을 찾을 수 없습니다.")
    print(f"사용 가능한 컬럼: {list(df.columns)}")
    print("\n아래 코드의 'name_column' 변수를 실제 관광지명 컬럼으로 수정하세요.")
else:
    # 클러스터 2 필터링
    cluster2_data = df[df['cluster'] == 2].copy()

    print(f"클러스터 2 전체 관광지 수: {len(cluster2_data)}")
    print(f"\n클러스터 2 전체 관광지 목록:")
    print(cluster2_data[name_column].tolist())

    # 특정 관광지만 필터링
    filtered_data = cluster2_data[cluster2_data[name_column].isin(target_spots)].copy()

    print(f"\n필터링된 관광지 수: {len(filtered_data)}")
    print(f"필터링된 관광지:")
    print(filtered_data[name_column].tolist())

    # 누락된 관광지 확인
    found_spots = filtered_data[name_column].tolist()
    missing_spots = [spot for spot in target_spots if spot not in found_spots]

    if missing_spots:
        print(f"\n⚠️  클러스터 2에서 찾을 수 없는 관광지:")
        for spot in missing_spots:
            print(f"  - {spot}")
        print("\n전체 데이터에서 검색 중...")

        # 전체 데이터에서 검색
        for spot in missing_spots:
            matches = df[df[name_column].str.contains(spot, na=False)]
            if len(matches) > 0:
                print(f"\n'{spot}' 검색 결과:")
                print(matches[[name_column, 'cluster', 'LATITUDE', 'LONGITUDE']].to_string(index=False))

    # GeoDataFrame 생성
    if len(filtered_data) > 0:
        gdf = gpd.GeoDataFrame(
            filtered_data,
            geometry=[Point(row['LONGITUDE'], row['LATITUDE']) for _, row in filtered_data.iterrows()],
            crs='EPSG:4326'
        )

        # GeoJSON 저장
        output_path = "C:/data/ulsan/cluster2_selected_spots.geojson"
        gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')

        print(f"\n✅ GeoJSON 파일 생성 완료:")
        print(f"   {output_path}")
        print(f"\n포함된 관광지: {len(filtered_data)}개")

        # 통계 정보 출력
        print("\n[선택된 관광지 통계]")
        print(f"평균 이용건수: {filtered_data['COUNT'].mean():.0f}")
        print(f"총 이용건수: {filtered_data['COUNT'].sum():.0f}")

    else:
        print("\n❌ 필터링된 데이터가 없습니다.")

# =====================================
# 부분 문자열 매칭 (관광지명이 정확히 일치하지 않을 경우)
# =====================================
print("\n" + "=" * 50)
print("부분 문자열 매칭 시도")
print("=" * 50)

# 클러스터 2 데이터
cluster2_data = df[df['cluster'] == 2].copy()

# 부분 문자열로 매칭
filtered_data_partial = cluster2_data[
    cluster2_data[name_column].str.contains('|'.join(target_spots), na=False, case=False)
].copy()

print(f"\n부분 매칭으로 찾은 관광지 수: {len(filtered_data_partial)}")

if len(filtered_data_partial) > 0:
    print("\n찾은 관광지:")
    print(filtered_data_partial[name_column].tolist())

    # GeoDataFrame 생성
    gdf_partial = gpd.GeoDataFrame(
        filtered_data_partial,
        geometry=[Point(row['LONGITUDE'], row['LATITUDE']) for _, row in filtered_data_partial.iterrows()],
        crs='EPSG:4326'
    )

    # GeoJSON 저장
    output_path_partial = "C:/Users/data/ulsan/cluster2_selected_spots_partial.geojson"
    gdf_partial.to_file(output_path_partial, driver='GeoJSON', encoding='utf-8')

    print(f"\n✅ GeoJSON 파일 생성 완료 (부분 매칭):")
    print(f"   {output_path_partial}")



    # =====================================
    # 클러스터 3 특정 관광지만 필터링
    # =====================================

    # 필터링할 관광지 목록
    target_spots = [
        '작수천',
        '자수정동굴나라',
        '신불산',
        '울주 언양읍성',
        '파래소폭포',
        '홍류폭포'
    ]

    # 데이터 로드 (이미 군집분석이 완료된 파일)
    try:
        df = ulsan_data1.copy()
    except NameError:
        # CSV 파일에서 로드
        df = pd.read_csv("C:/Users/data/ulsan/ulsan_tour_clusters.csv", encoding='utf-8-sig')

    # 관광지명 컬럼 (실제 컬럼명에 맞게 수정)
    name_column = 'TRANM_x'  # ← 실제 관광지명 컬럼으로 변경

    # 컬럼 존재 확인
    if name_column not in df.columns:
        print(f"⚠️  '{name_column}' 컬럼을 찾을 수 없습니다.")
        print(f"사용 가능한 컬럼: {list(df.columns)}")
        print("\n아래 코드의 'name_column' 변수를 실제 관광지명 컬럼으로 수정하세요.")
    else:
        # 클러스터 3 필터링
        cluster3_data = df[df['cluster'] == 3].copy()

        print("=" * 50)
        print(f"클러스터 3 분석")
        print("=" * 50)
        print(f"클러스터 3 전체 관광지 수: {len(cluster3_data)}")
        print(f"\n클러스터 3 전체 관광지 목록:")
        for name in cluster3_data[name_column].tolist():
            print(f"  - {name}")

        # 특정 관광지만 필터링 (정확한 매칭)
        filtered_data = cluster3_data[cluster3_data[name_column].isin(target_spots)].copy()

        print(f"\n{'=' * 50}")
        print(f"정확한 이름 매칭 결과")
        print(f"{'=' * 50}")
        print(f"필터링된 관광지 수: {len(filtered_data)}")

        if len(filtered_data) > 0:
            print(f"\n✅ 찾은 관광지:")
            for name in filtered_data[name_column].tolist():
                print(f"  - {name}")

        # 누락된 관광지 확인
        found_spots = filtered_data[name_column].tolist()
        missing_spots = [spot for spot in target_spots if spot not in found_spots]

        if missing_spots:
            print(f"\n⚠️  정확히 일치하는 이름을 찾을 수 없는 관광지:")
            for spot in missing_spots:
                print(f"  - {spot}")

        # 부분 문자열 매칭
        print(f"\n{'=' * 50}")
        print(f"부분 문자열 매칭 시도")
        print(f"{'=' * 50}")

        filtered_data_partial = pd.DataFrame()

        for target in target_spots:
            # 각 키워드별로 부분 매칭
            matches = cluster3_data[
                cluster3_data[name_column].str.contains(target, na=False, case=False)
            ]

            if len(matches) > 0:
                print(f"\n'{target}' 검색 결과:")
                for name in matches[name_column].tolist():
                    print(f"  → {name}")
                filtered_data_partial = pd.concat([filtered_data_partial, matches])

        # 중복 제거
        filtered_data_partial = filtered_data_partial.drop_duplicates()

        print(f"\n{'=' * 50}")
        print(f"최종 결과")
        print(f"{'=' * 50}")
        print(f"부분 매칭으로 찾은 관광지 수: {len(filtered_data_partial)}")

        if len(filtered_data_partial) > 0:
            print("\n최종 선택된 관광지:")
            for idx, row in filtered_data_partial.iterrows():
                print(f"  - {row[name_column]} (이용건수: {row['COUNT']:,})")

            # GeoDataFrame 생성
            gdf = gpd.GeoDataFrame(
                filtered_data_partial,
                geometry=[Point(row['LONGITUDE'], row['LATITUDE']) for _, row in filtered_data_partial.iterrows()],
                crs='EPSG:4326'
            )

            # GeoJSON 저장
            output_path = "C:/Users/data/ulsan/cluster3_selected_spots.geojson"
            gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')

            print(f"\n✅ GeoJSON 파일 생성 완료:")
            print(f"   {output_path}")

            # 통계 정보
            print(f"\n[선택된 관광지 통계]")
            print(f"총 관광지 수: {len(filtered_data_partial)}")
            print(f"평균 이용건수: {filtered_data_partial['COUNT'].mean():.0f}")
            print(f"총 이용건수: {filtered_data_partial['COUNT'].sum():.0f}")
            print(f"최대 이용건수: {filtered_data_partial['COUNT'].max():.0f}")
            print(f"최소 이용건수: {filtered_data_partial['COUNT'].min():.0f}")

        else:
            print("\n❌ 필터링된 데이터가 없습니다.")
            print("\n전체 데이터에서 검색 중...")

            # 전체 데이터에서 검색
            for spot in target_spots:
                print(f"\n'{spot}' 전체 검색:")
                matches = df[df[name_column].str.contains(spot, na=False, case=False)]
                if len(matches) > 0:
                    for _, row in matches.iterrows():
                        print(
                            f"  → {row[name_column]} (클러스터: {row['cluster']}, 위도: {row['LATITUDE']}, 경도: {row['LONGITUDE']})")
                else:
                    print(f"  → 찾을 수 없음")

    # =====================================
    # 클러스터 3 전체 관광지 목록 CSV 저장
    # =====================================
    try:
        cluster3_full = df[df['cluster'] == 3].copy()
        output_csv = "C:/Users/data/ulsan/cluster3_all_spots.csv"
        cluster3_full.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n📄 클러스터 3 전체 목록 CSV 저장:")
        print(f"   {output_csv}")
    except:
        pass

# =====================================
# 클러스터 5 특정 관광지만 필터링
# =====================================

# 필터링할 관광지 목록
target_spots = [
    '울산대곡박물관',
    '울산암각화박물관',
    '충렬공박제상유적지',
    '울주 대곡리 반구대암각화',
    '울주 천전리 각석',
    '울산어린이천문대'
]

# 데이터 로드 (이미 군집분석이 완료된 파일)
try:
    df = ulsan_data1.copy()
except NameError:
    # CSV 파일에서 로드
    df = pd.read_csv("C:/Users/data/ulsan/ulsan_tour_clusters.csv", encoding='utf-8-sig')

# 관광지명 컬럼 (실제 컬럼명에 맞게 수정)
name_column = 'TRANM_x'  # ← 실제 관광지명 컬럼으로 변경

# 컬럼 존재 확인
if name_column not in df.columns:
    print(f"⚠️  '{name_column}' 컬럼을 찾을 수 없습니다.")
    print(f"사용 가능한 컬럼: {list(df.columns)}")
    print("\n아래 코드의 'name_column' 변수를 실제 관광지명 컬럼으로 수정하세요.")
else:
    # 클러스터 5 필터링
    cluster5_data = df[df['cluster'] == 5].copy()

    print("=" * 50)
    print(f"클러스터 5 분석")
    print("=" * 50)
    print(f"클러스터 5 전체 관광지 수: {len(cluster5_data)}")
    print(f"\n클러스터 5 전체 관광지 목록:")
    for name in cluster5_data[name_column].tolist():
        print(f"  - {name}")

    # 특정 관광지만 필터링 (정확한 매칭)
    filtered_data = cluster5_data[cluster5_data[name_column].isin(target_spots)].copy()

    print(f"\n{'=' * 50}")
    print(f"정확한 이름 매칭 결과")
    print(f"{'=' * 50}")
    print(f"필터링된 관광지 수: {len(filtered_data)}")

    if len(filtered_data) > 0:
        print(f"\n✅ 찾은 관광지:")
        for name in filtered_data[name_column].tolist():
            print(f"  - {name}")

    # 누락된 관광지 확인
    found_spots = filtered_data[name_column].tolist()
    missing_spots = [spot for spot in target_spots if spot not in found_spots]

    if missing_spots:
        print(f"\n⚠️  정확히 일치하는 이름을 찾을 수 없는 관광지:")
        for spot in missing_spots:
            print(f"  - {spot}")

    # 부분 문자열 매칭
    print(f"\n{'=' * 50}")
    print(f"부분 문자열 매칭 시도")
    print(f"{'=' * 50}")

    filtered_data_partial = pd.DataFrame()

    # 각 키워드별 검색 (더 정교한 매칭)
    search_keywords = {
        '울산대곡박물관': ['울산대곡박물관'],
        '울산암각화박물관': ['울산암각화박물관'],
        '충렬공박제상유적지': ['충렬공박제상유적지'],
        '울주 대곡리 반구대암각화': ['울주 대곡리 반구대암각화'],
        '울주 천전리 각석': ['울주 천전리 각석'],
        '울산어린이천문대': ['울산어린이천문대']
    }

    for target, keywords in search_keywords.items():
        found = False
        for keyword in keywords:
            matches = cluster5_data[
                cluster5_data[name_column].str.contains(keyword, na=False, case=False)
            ]

            if len(matches) > 0 and not found:
                print(f"\n'{target}' 검색 결과 (키워드: '{keyword}'):")
                for name in matches[name_column].tolist():
                    print(f"  → {name}")
                filtered_data_partial = pd.concat([filtered_data_partial, matches])
                found = True
                break

        if not found:
            print(f"\n'{target}' → 찾을 수 없음")

    # 중복 제거
    filtered_data_partial = filtered_data_partial.drop_duplicates()

    print(f"\n{'=' * 50}")
    print(f"최종 결과")
    print(f"{'=' * 50}")
    print(f"부분 매칭으로 찾은 관광지 수: {len(filtered_data_partial)}")

    if len(filtered_data_partial) > 0:
        print("\n최종 선택된 관광지:")
        for idx, row in filtered_data_partial.iterrows():
            print(f"  - {row[name_column]} (이용건수: {row['COUNT']:,})")

        # GeoDataFrame 생성
        gdf = gpd.GeoDataFrame(
            filtered_data_partial,
            geometry=[Point(row['LONGITUDE'], row['LATITUDE']) for _, row in filtered_data_partial.iterrows()],
            crs='EPSG:4326'
        )

        # GeoJSON 저장
        output_path = "C:/Users/data/ulsan/cluster5_selected_spots.geojson"
        gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')

        print(f"\n✅ GeoJSON 파일 생성 완료:")
        print(f"   {output_path}")

        # 통계 정보
        print(f"\n[선택된 관광지 통계]")
        print(f"총 관광지 수: {len(filtered_data_partial)}")
        print(f"평균 이용건수: {filtered_data_partial['COUNT'].mean():.0f}")
        print(f"총 이용건수: {filtered_data_partial['COUNT'].sum():.0f}")
        print(f"최대 이용건수: {filtered_data_partial['COUNT'].max():.0f}")
        print(f"최소 이용건수: {filtered_data_partial['COUNT'].min():.0f}")

        # 관광지별 상세 정보
        print(f"\n[관광지별 상세 정보]")
        for idx, row in filtered_data_partial.iterrows():
            print(f"\n{row[name_column]}")
            print(f"  - 위도: {row['LATITUDE']:.6f}")
            print(f"  - 경도: {row['LONGITUDE']:.6f}")
            print(f"  - 이용건수: {row['COUNT']:,}")
            print(f"  - 정규화값: {row['Value']:.3f}")

    else:
        print("\n❌ 클러스터 5에서 필터링된 데이터가 없습니다.")
        print("\n전체 데이터에서 검색 중...")

        # 전체 데이터에서 검색
        for spot in target_spots:
            print(f"\n'{spot}' 전체 검색:")
            # 키워드 추출 (공백 기준 분리)
            keywords = spot.split()
            found_any = False

            for keyword in keywords:
                if len(keyword) > 1:  # 1글자 키워드 제외
                    matches = df[df[name_column].str.contains(keyword, na=False, case=False)]
                    if len(matches) > 0:
                        found_any = True
                        for _, row in matches.iterrows():
                            print(f"  → {row[name_column]} (클러스터: {row['cluster']}, 이용건수: {row['COUNT']:,})")

            if not found_any:
                print(f"  → 전체 데이터에서도 찾을 수 없음")

# =====================================
# 클러스터 5 전체 관광지 목록 CSV 저장
# =====================================
try:
    cluster5_full = df[df['cluster'] == 5].copy()
    output_csv = "C:/Users/Desktop/data/ulsan/cluster5_all_spots.csv"
    cluster5_full.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n📄 클러스터 5 전체 목록 CSV 저장:")
    print(f"   {output_csv}")
    print(f"   (총 {len(cluster5_full)}개 관광지)")
except:
    pass

print("\n" + "=" * 50)
print("작업 완료!")
print("=" * 50)