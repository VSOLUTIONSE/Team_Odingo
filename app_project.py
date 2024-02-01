import os
from langchain.tools import DuckDuckGoSearchRun
from crewai import Agent, Crew, Task, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# instantiating Gemini Pro as LLM

gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                    verbose=True,
                                    temperature=0.5,
                                    google_api_key="AIzaSyDLBu2I_Fw9aoUB5sL3XOidQEUr4ivXwM4")


# creates Search
search_tool= DuckDuckGoSearchRun()


# Defining the Agents

Researcher =Agent(
    role="Astute Researcher",
    goal="Make an educative research and key highlights in Generative AI",
    backstory="Seasoned researcher with proven track record of excellent research in Generative AI. Highly skilled in easily articulating complex concept to engage diverse audience",
    verbose=True,               #Logging extensive output detail
    allow_delegation= False,    #enable collaboration between agents
    tools=[search_tool],        #tools used
    llm= gemini_llm             #llm as gemini_llm

)

Writer = Agent(
    role="Prolific Writer",
    goal="Uncover and Compose educative content about Generative AI",
    backstory="You are a dedicated and experienced content strategist with proven track record in crafting piece that facilitates learning, and making complex tech topics easy",
    verbose=True,
    allow_delegation=True,
    llm = gemini_llm  
)

Examiner = Agent(
    role= "Skilled Examiner",
    goal="craft engaging questions to test and assess the understanding of a newbie or student, along side the correct anwser",
    backstory=" you are a seasoned examiner and educator with a proven track record of excellence in examining and teaching. With a passion for fostering inclusive learning and evaluating the understanding of students or newbies through quality tests.",
    verbose=True,
    allow_delegation=True, 
    llm = gemini_llm   
)

# Defining Unique Task for each Agents

task1 = Task(
    description="Craft a detailed research in Generative AI thats educative",
    agent= Researcher
)
task2 = Task(
    description="Craft an informative piece, that's engaging and educative, explaining and describing Generative AI.",
    agent= Writer
)
task3 = Task(
    description="Formulate 3 test questions to evaluate how well the subject on Generative AI has been well understood by the newbie or student, provide answers to this questions alongside",
    agent= Examiner
)

# Instantiatiate the Crew/Team of Agents

crew = Crew(
    agents=[Researcher,Writer,Examiner],
    tasks=[task1,task2,task3],
    verbose= 2,
    process= Process.sequential,

)

result= crew.kickoff()

print("##################################")
print(result)

