import json

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)


# from langchain_community.llms.ollama import Ollama

# m = Ollama(model="starcoder2:3b")
# r = m.generate(prompts=['Give me the name of the one llm model.']*2)
# # print(r)
# print(r.generations[0][0].text)
# print(r.generations[1][0].text)