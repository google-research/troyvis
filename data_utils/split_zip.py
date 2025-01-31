import os
import zipfile

def create_zip_part(folder_path, part_number, files):
    part_filename = f"part_{part_number}.zip"
    with zipfile.ZipFile(part_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            arcname = os.path.relpath(file, folder_path)
            zipf.write(file, arcname=arcname)
    return part_filename

def split_folder_into_zip_parts(folder_path, part_size_limit):
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    part_number = 0
    part_files = []
    part_size = 0
    
    for file in all_files:
        file_size = os.path.getsize(file)
        if part_size + file_size > part_size_limit:
            if part_files:
                part_number += 1
                create_zip_part(folder_path, part_number, part_files)
                part_files = []
                part_size = 0
        part_files.append(file)
        part_size += file_size

    # Create the last part if there are remaining files
    if part_files:
        part_number += 1
        create_zip_part(folder_path, part_number, part_files)

folder_path = "path_to_your_folder"
part_size_limit = 5 * 1024 * 1024 * 1024  # 5GB
split_folder_into_zip_parts(folder_path, part_size_limit)
