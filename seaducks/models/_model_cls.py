import pickle

class Model:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # read-only properties
    @property
    def string_name(self):
        pass
    @property
    def model(self):
        pass
    
    def fit(self):
        pass

    def save_model(self, file_name):
        filehandler = open(f"{file_name}.p","wb")
        pickle.dump(self,filehandler)