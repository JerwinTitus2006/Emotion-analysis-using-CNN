import csv
import os

CSV_FILE = "detections.csv"

def log_detection(timestamp, gender, age, emotion):
    write_header = not os.path.exists(CSV_FILE)

    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["timestamp", "gender", "age", "emotion"])
        writer.writerow([timestamp, gender, age, emotion])
