
import csv

class Metadata:

    __metadata_path = ""
    __metadata = []

    def __init__(self,path):
        self.__metadata_path = path

    #load metadata from CSV, call it before any other methods
    def load(self):
        with open(self.__metadata_path, newline='') as metadatacsv:
            reader = csv.DictReader(metadatacsv)
            for row in reader:
                self.__metadata.append(row)

    #testing purpose only 
    def print(self):
        for row in self.__metadata:
            print(row)