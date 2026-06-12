#execution.nlp_intent.intent_engine.py

from nlp.label_mapper import LabelMapper
from core.contracts.scientific_intent import ScientificIntent
from using_libraries.prediction_to_scientific_intent_converter import PredictionToScientificIntentConverter
from using_libraries.parameter_sanitizer import ParameterSanitizer
from using_libraries.symbolic_extractor import SymbolicExtractor
from using_libraries.predictions_merger import merge_predictions

class IntentEngine:

    def __init__(self, intent_detector, slot_generator=None, converter=None, sanitizer=None):        
        
        if intent_detector is None:
            raise ValueError(
                "IntentEngine requires a BaseIntentDetector instance"
            )

        self.intent_detector = intent_detector
        self.slot_generator = slot_generator    #Legacy Code

        self.converter = (
            PredictionToScientificIntentConverter()
        )

        self.sanitizer = (
            ParameterSanitizer()
        )
        
        self.symbolic_extractor = (
            SymbolicExtractor()
            )

    def detect(self, query):

        neural_prediction = self.intent_detector.detect(query)

        symbolic_slots = self.symbolic_extractor.extract(query)

        merged = merge_predictions(neural_prediction, symbolic_slots)

        intent = self.converter.convert(merged)

        intent = self.sanitizer.sanitize(intent)

        return intent