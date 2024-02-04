import os

def count_files_in_folder(folder_path):
    try:
        # Get the list of files in the specified folder
        files = os.listdir(folder_path)

        # Count the number of files
        file_count = len(files)

        # Print or return the result
        print(f'The number of files in {folder_path} is: {file_count}')
        return file_count

    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
        return 0
    

def check_file_exists(file_path):
    return os.path.exists(file_path)
