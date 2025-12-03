import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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
    '나사해수욕장',
    '솔개해수욕장',
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
name_column = 'NAME'  # ← 실제 관광지명 컬럼으로 변경

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
        output_path = "C:/Users/data/ulsan/cluster2_selected_spots.geojson"
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