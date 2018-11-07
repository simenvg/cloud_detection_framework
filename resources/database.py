import sqlite3


def initialize_database():
    conn = db.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE detections
                         (image_name text, xmin integer, xmax integer, ymin integer, ymax integer, class_name text, confidence real)''')
    return conn


def add_to_db(conn, image_name, xmin, xmax, ymin, ymax, class_name, confidence):
    c = conn.cursor()
    c.execute("INSERT INTO detections (image_name, xmin, xmax, ymin, ymax, class_name, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)"), (
        image_name, xmin, xmax, ymin, ymax, class_name, confidence)
