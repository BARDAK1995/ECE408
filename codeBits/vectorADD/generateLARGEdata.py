import random

num_lines = 80000000
file_path = "large_input.raw"

# Open the file in write mode
with open(file_path, "w") as file:
    # Write the number of lines at the top of the file
    file.write(f"{num_lines}\n")
    
    # Generate num_lines random floating point numbers and write them to the file
    for _ in range(num_lines):
        random_num = random.uniform(0, 100)  # Generate a random number between 0 and 100
        file.write(f"{random_num:.2f}\n")  # Write the number to the file with 2 decimal places
