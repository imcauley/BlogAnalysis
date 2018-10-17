import os
import xml.etree.ElementTree as ET
import re

def get_data():
    path = "./blogs/"

    file_list = os.listdir(path)

    total_data = []
    for file_name in file_list:
        data = get_file_data(path + file_name)

        total_data += data

    return total_data

def get_gender(file_name):
    elements = file_name.split('.')
    gender = elements[2]

    return gender

def get_file_data(file_name):
    gender = get_gender(file_name)

    with open(file_name, 'rb') as f:
        text = f.read()
        text = preprocess_text(text)
        document = parse_document(text)

    data = [(post, gender) for post in document]

    return data

def preprocess_text(text):
    return text.decode('utf-8', errors='replace')
def parse_document(text):
    doc = re.finditer(r"\<post\>\s*(.*)\s*\<\/post\>", text)

    doc = [d.group(1) for d in doc]

    return doc


if __name__ == "__main__":
    print(len(get_data()))
    # print(get_file_data("./blogs/1755542.female.24.indUnk.Scorpio.xml"))