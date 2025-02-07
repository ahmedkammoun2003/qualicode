import os
import pickle as pkl
import numpy as np
from typing import Dict

# A dictionary to hold the loaded objects
loaded_objects = {}

def sort_by_number(filename):
    number = filename.split('.')[0]
    try:
        return int(number)
    except ValueError:
        return float('inf')  # Place non-integer filenames at the end of the sorted list

def deal_result(root_path):
    filename_list = os.listdir(root_path)
    filename_list = sorted(filename_list, key=sort_by_number)
    save_result = {}
    for filename in filename_list:
        try:
            with open(os.path.join(root_path, filename), 'rb') as f:
                data = pkl.load(f)
            file_id = int(filename.split('.')[0])
            result = data[file_id]["results"]
            print(f"Loaded results for file_id {file_id}: {result}")  # Debugging line
            save_result[file_id] = result
        except Exception as e:
            print(f"Error loading {filename}: {e}")  # Print the error for debugging
    return save_result

def persistent_load(persistent_id):
    # Check if the persistent_id is already loaded
    if persistent_id in loaded_objects:
        return loaded_objects[persistent_id]
    
    # Logic to load the object based on the persistent_id
    # For example, if persistent_id is a tuple (type, id):
    obj_type, obj_id = persistent_id
    
    # Here you would implement the logic to retrieve the object based on its type and ID
    # This is a placeholder; you need to replace it with your actual loading logic
    if obj_type == 'some_type':  # Replace 'some_type' with actual type
        # Load or create the object as needed
        obj = load_some_type(obj_id)  # Implement this function based on your needs
        loaded_objects[persistent_id] = obj  # Cache the loaded object
        return obj
    
    raise ValueError(f"Unknown persistent_id: {persistent_id}")

def load_some_type(obj_id):
    # Implement the logic to load or create the object based on obj_id
    # This is just a placeholder function
    return None  # Replace with actual object loading logic

def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:  # Open the file in binary read mode
            unpickler = pkl.Unpickler(file)
            unpickler.persistent_load = persistent_load  # Set the persistent load function
            data = unpickler.load()  # Load the data from the pickle file
            return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except pkl.UnpicklingError:
        print(f"Error: The file {file_path} is not a valid pickle file.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_results(results: Dict[int, list], count_errors: bool = False, k_list: list = [1, 10, 100]):
    if not results:
        raise ValueError("Results are empty. Please check the input data.")
    
    # Existing logic follows...
