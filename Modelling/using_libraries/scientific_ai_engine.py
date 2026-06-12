#using_libraries\scientific_ai_engine.py

import torch
import json
from transformers import AutoTokenizer

from core.router.scientific_router import ScientificRouter
from execution.intent_execution_bridge import IntentExecutionBridge
from registrys.executor_registry import ExecutorRegistry, build_registry
from execution.executors.hydrogen_executor import HydrogenDomainExecutor
from using_libraries.intent_training.lib_train_serialization import save_checkpoint, load_model
from nlp.detectors.neural_intent_adapter import NeuralIntentAdapter
from execution.nlp_intent.intent_engine import IntentEngine



def _load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

class ScientificAIEngine:

    def __init__(self, model_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model, self.stats, \
        self.model_config, \
        self.domain_action_modes_config = load_model(
            model_path,
            self.device
        )
        
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.neural_adapter = NeuralIntentAdapter(
            model=self.model,
            tokenizer=self.tokenizer,
            stats=self.stats,
            config=self.domain_action_modes_config,
            device=self.device
        )

        self.intent_engine = IntentEngine(
            intent_detector=self.neural_adapter
        )
        
        self.registry = build_registry()

        self.execution_bridge = IntentExecutionBridge(
            self.registry
        )

        self.router = ScientificRouter(
            self.intent_engine,
            self.execution_bridge
        )

    def run(self, query):

        return self.router.route(query)

    def detect(self, query):

        return self.intent_engine.detect(query)
