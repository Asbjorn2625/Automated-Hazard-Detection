import os
import re

directory = "./Label_size_test"  # Replace with the path to your folder
order_map = {
    0: 1,
    10: 2,
    20: 3,
    30: 4,
    45: 5,
    -10: 6,
    -20: 7,
    -30: 8,
    -45: 9
}

files = sorted(os.listdir(directory))

angles = [0, 10, 20, 30, 45, -10, -20, -30, -45]

def get_set_number(num):
    return num // len(angles) + 2

for file in files:
    match = re.match(r'(depth|rgb)_image_(\d{4})\.(raw|png)', file)
    if match:
        prefix, num, extension = match.groups()
        num = int(num) - 1  # Adjust for the starting index at 0001
        angle = angles[num % len(angles)]
        set_number = get_set_number(num)

        new_name = f'{prefix}{set_number}_{angle}.{extension}'
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)