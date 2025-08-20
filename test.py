from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.agents import Tool, AgentType, initialize_agent
import os
import config

# --- Optional tracing ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "..."


# 1. Load Gemma-3-270M-IT model and tokenizer
model_name = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")  # load to GPU or CPU
#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, 
#                    max_new_tokens=256, temperature=0)
end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
eos_id = end_id if isinstance(end_id, int) and end_id >= 0 else tokenizer.eos_token_id
model.generation_config.eos_token_id = eos_id
model.generation_config.pad_token_id = eos_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    #clean_up_tokenization_spaces=True,
    max_new_tokens=256,
    #do_sample=False,
    temperature=0.0,
)

#llm = HuggingFacePipeline(pipeline=pipe)  # LangChain wrapper for the pipeline
llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

# 2. Define a simple tool (adds two numbers from input string)
def get_name(person: str) -> str:
    print(f"========>{person}<=========")
    return f"Name of the {person} is Fischer!"
    #import re
    #nums = [int(x) for x in re.findall(r"\d+", query)]
    #return str(sum(nums) + 23)

add_tool = Tool(name="Intermediate Answer", func=get_name, 
                description="Returns the name of a person.\n Agr: person: description of a person to get name of")

# 3. Initialize a ReAct agent with the Gemma LLM and tool
agent = initialize_agent(tools=[add_tool], llm=llm, 
                         agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True, 
                         handle_parsing_errors=True,
                         max_iterations=3           
)

# 4. Run the agent on a query that requires the tool
result = agent.invoke("What is the name of first person on Jupiter? Use the 'Intermediate Answer' if needed.")
print(result)