import os
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew
from langchain import Ollama

load_dotenv() 

# Using CHAT GPT-4 or Serper
# os.environ['SERPER_API_KEY']
api = os.environ['OPEN_API_KEY']

# Using Ollama Local model
ollama = Ollama(model="openhermes")

researcher = Agent(
    role="Researcher",
    description="Researches the market",
    goal="Research the market to find best items to write about",
    backstory="You are a researcher",
    verbose=True,
    allow_delegation=True,
    llm="ollama"
)

writer = Agent(
    role="Writer",
    description="Writes articles",
    goal="Write articles about hot topics",
    backstory="You are a writer",
    verbose=True,
    allow_delegation=True,
    llm="ollama"
)

research = Task(
    description="Investigate market with hot topics",
    agent=researcher,
)

write = Task(
    description="Write articles about hot topics",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=2,
    process=Process.sequential
)

print("############")
result = crew.kickoff()