# coding=utf-8
import requests
import sys

if __name__ == '__main__':
    url = 'http://127.0.0.1:8080/inference'
    # url = 'http://172.18.160.172:8080/inference'
    file_path = '/home/dimitri/Pictures/daisy.jpg'
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    if len(sys.argv) > 2:
        url = sys.argv[2]

    with open(file_path, 'rb') as f:
        image = {'image': f}
        res = requests.post(url, files=image)
        print(res.text)
