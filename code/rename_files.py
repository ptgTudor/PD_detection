import os

def rename_files(folder):
    for filename in os.listdir(folder):
        # Check if the filename has at least 10 characters to avoid index error
        if len(filename) >= 10:
            # Insert 'P' after the first 9 characters
            new_name = filename[:9] + 'P' + filename[9:]
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_name}'")
        else:
            print(f"Skipped '{filename}' as it does not meet the length criteria")

# Replace 'path/to/your/folder' with the actual path to your folder
folder_path = 'F:\programare\project code\data\PC-GITA_per_task_44100Hz\_vowels\_normalized\U\dataset_output\pd_output'
rename_files(folder_path)
