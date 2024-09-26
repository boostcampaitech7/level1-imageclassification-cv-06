import os
import shutil

# 폴더 경로 설정
source_dir = './data/converted_images'
destination_dir = './data/train'

# 하위 폴더 구조 유지하며 파일 이동
for root, dirs, files in os.walk(source_dir):
    # 현재 디렉토리와 destination_dir에 대응되는 경로 생성
    relative_path = os.path.relpath(root, source_dir)
    dest_path = os.path.join(destination_dir, relative_path)

    # 대상 폴더가 없으면 생성
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # 파일 이동
    for file in files:
        source_file = os.path.join(root, file)
        dest_file = os.path.join(dest_path, file)

        # 파일을 이동하거나 복사 (shutil.move를 shutil.copy로 변경 가능)
        shutil.move(source_file, dest_file)

print("모든 파일이 이동되었습니다.")
