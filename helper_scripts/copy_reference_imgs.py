import os
import csv

images = os.listdir("D:/OneDrive - Venusnet/Dokumente/9. Record Bin/1. Archiv/3. KSSO/Matura/Maturarbeit/Golf_course_IMGs")

with open("Dataset.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        source_id = row[0]
        for img in images:
            # copy reference image
            if source_id in img:
                source_path = os.path.join("D:/OneDrive - Venusnet/Dokumente/9. Record Bin/1. Archiv/3. KSSO/Matura/Maturarbeit/Golf_course_IMGs", img)
                destination_path = os.path.join("reference_imgs", img)
                os.system(f'copy "{source_path}" "{destination_path}"')
                