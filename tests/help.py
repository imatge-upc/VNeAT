import csv
from collections import OrderedDict



class DataObject():

    def __init__(self,id):
        self.id = id
        self.x = None
        self.y = None
        self.month =None
        self.day = None
        self.FFMC = None
        self.DMC = None
        self.DC = None
        self.ISI =None
        self.temp = None
        self.RH = None
        self.wind = None
        self.rain = None
        self.area = None




class Loader():

    @staticmethod
    def load_db():
        clinical_path = 'C:\\Users\\upcnet\\Datasets\\ForestFires\\forestfires.csv'
        data=OrderedDict()

        with open(clinical_path) as clinical_file:
            reader = csv.DictReader(clinical_file)
            for row in reader:
                data[reader.line_num]=DataObject(reader.line_num)
                data[reader.line_num].x = row['X']
                data[reader.line_num].y = row['Y']
                data[reader.line_num].month = row['month']
                data[reader.line_num].day = row['day']
                data[reader.line_num].FFMC = row['FFMC']
                data[reader.line_num].DMC = row['DMC']
                data[reader.line_num].DC = row['DC']
                data[reader.line_num].ISI = row['ISI']
                data[reader.line_num].temp = row['temp']
                data[reader.line_num].RH = row['RH']
                data[reader.line_num].wind = row['wind']
                data[reader.line_num].rain = row['rain']
                data[reader.line_num].area = row['area']

        return data