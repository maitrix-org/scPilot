from abc import ABC, abstractmethod

class BaseEvaluationAgent(ABC):
    def __init__(self, hypothesis, output_dir):
        self.hypothesis = hypothesis
        self.output_dir = output_dir
        self.evaluation_results = None
        
    @abstractmethod
    def evaluate(self):
        pass
    
    def get_evaluation_results(self):
        return self.evaluation_results