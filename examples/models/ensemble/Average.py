from rafiki.model import BaseModel, InvalidModelParamsException, validate_model_class
from rafiki.constants import TaskType

class Average(BaseModel):

    def get_knob_config(self):
        return {
            'knobs': {}
        }

    def init(self, knobs):
        pass

    def train(self, dataset_uri):
        pass

    def evaluate(self, dataset_uri):
        pass

    def predict(self, queries):
        pass

    def destroy(self):
        pass

    def dump_parameters(self):
        pass

    def load_parameters(self, params):
        pass