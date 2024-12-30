import pickle

class Model:
    def __init__(self):
        pass

    def save_model(self, file_name):
        filehandler = open(f"{file_name}.p","wb")
        pickle.dump(self,filehandler)