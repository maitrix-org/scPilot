from abc import ABC, abstractmethod

class BaseEnvironmentAgent(ABC):
    def __init__(self, simulation_environment, input_dir, output_dir):
        self.simulation_environment = simulation_environment
        self.input_dir = input_dir
        self.output_dir = output_dir
