import os
import shutil

def move_all_files(src_folder, dest_folder):
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Walk through the source directory
    for root, _, files in os.walk(src_folder):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_folder, file)

            # Ensure no overwriting: rename if needed
            base, extension = os.path.splitext(file)
            counter = 1
            while os.path.exists(dest_path):
                new_file_name = f"{base}_{counter}{extension}"
                dest_path = os.path.join(dest_folder, new_file_name)
                counter += 1

            shutil.move(src_path, dest_path)
            print(f"Moved: {src_path} -> {dest_path}")

# Example usage
source = 'E:\datasets_box'
destination = 'real_datasets_box'
move_all_files(source, destination)
