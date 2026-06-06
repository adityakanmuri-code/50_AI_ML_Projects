import yaml
import os

class Config:
    def __init__(self):
        config_path = os.path.join(os.getcwd(),"Spam_Ham_Model_RNN_Word2Vec\\config\\config.yaml")
        with open(config_path,"r") as file:
            self.config =yaml.safe_load(file)
    
    def get(self,*keys):
        value = self.config
        for key in keys:
            value = value[key]
        return value