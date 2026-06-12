#nlp\detectors\neural_intent_adapter.py


from nlp.detectors.base_intent_detector import BaseIntentDetector
from using_libraries.intent_training.predict_inference import parse_query


class NeuralIntentAdapter(BaseIntentDetector):

    def __init__(self,
        model,
        tokenizer,
        stats,
        config,
        device):

        self.model = model
        self.tokenizer = tokenizer
        self.stats = stats
        self.config = config
        self.device = device


    def detect(self, query):

        return parse_query(
            query,
            self.model,
            self.tokenizer,
            self.stats,
            self.config,
            self.device
        )

####################   Legacy Code ############
# from nlp.detectors.base_intent_detector import BaseIntentDetector

# class NeuralIntentAdapter(BaseIntentDetector):

#     def __init__(self, model):

#         self.model = model

#     def detect(self, query):

#         return self.model.predict(query)
