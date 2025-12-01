from abc import ABC, abstractmethod
from openai import OpenAI
from config.settings import OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


class BaseHypothesisAgent(ABC):
    def __init__(self, hypothesis):
        self.hypothesis = hypothesis
        self.search_query = None
        self.literature = None
        self.summary = None
        self.refined_hypothesis = None

    def summarize_literature(self):
        formatted_lit = '\n'.join([f"{item['title']}: {item['link']}" for item in self.literature])
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in scientific literature summarization. Provide a concise summary of the key findings and insights from the given literature."},
            {"role": "user", "content": f"Here is the list of literature:\n\n{formatted_lit}\n\nPlease summarize the key findings and insights from these papers."}
        ],
        temperature=0.7)
        self.summary = response.choices[0].message.content.strip()

    @abstractmethod
    def refine_hypothesis(self, evaluation_result=None):
        pass

    def get_refined_hypothesis(self):
        return self.refined_hypothesis