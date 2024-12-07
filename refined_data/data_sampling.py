import pandas as pd

# 원본 CSV 파일 읽기
input_file = 'refined_data/completed_output_pair_A.csv'
output_file = 'refined_data/sub_sample.csv'  # 출력 파일 이름

# CSV 파일 로드
df = pd.read_csv(input_file)

# 랜덤하게 100개의 행 선택
sampled_df = df.sample(n=100, random_state=42)  # random_state로 결과 고정 가능

# 선택한 데이터 새로운 CSV 파일로 저장
sampled_df.to_csv(output_file, index=False)

print(f"랜덤으로 선택된 100개의 행이 {output_file}에 저장되었습니다.")
