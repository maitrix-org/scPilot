from abc import ABC, abstractmethod

class BaseExperimentAgent(ABC):
    def __init__(self, hypothesis, literature_summary=None):
        self.hypothesis = hypothesis
        self.literature_summary = literature_summary
        self.experiment_proposal = None
        
    @abstractmethod
    def propose_experiment(self):
        pass
    
    def get_experiment_proposal(self):
        return self.experiment_proposal
    
    def get_literature_summary(self):
        return self.literature_summary