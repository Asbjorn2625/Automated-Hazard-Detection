import os

base_filename = "rgb_image_"
output_folder = "image_name/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, "filenames.txt")

k = 4

with open(output_file, "w") as f:
    for i in range(1, 200):  # change range() arguments as needed
        if i == 10:
            k == 3
        if i == 100:
            k == 2
        filename = base_filename + str(i).zfill(k)
        f.write(filename + "\n")