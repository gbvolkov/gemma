import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# --- Optional tracing ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "..."

# --- Your single tool (must be named EXACTLY "search_kb") ---
from retrievers.retriever import get_search_tool
search_kb = get_search_tool()
assert getattr(search_kb, "name", "") == "search_kb", "Tool must be named 'search_kb'"

# =========================
# 1) Model (CPU-safe) setup
# =========================
MODEL_ID = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="float32",   # CPU-safe
    device_map="cpu",
)

# End-of-turn stop so the chat wrapper doesn't echo the prompt forever
end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
eos_id = end_id if isinstance(end_id, int) and end_id >= 0 else tokenizer.eos_token_id
model.generation_config.eos_token_id = eos_id
model.generation_config.pad_token_id = eos_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    clean_up_tokenization_spaces=True,
    max_new_tokens=160,
    do_sample=False,
    temperature=0.0,
)

llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

# =========================
# 2) Prompt designed for tool calling (ReAct)
# =========================
# IMPORTANT: The prebuilt ReAct agent expects EXACT lines:
#   Action: <tool_name>
#   Action Input: "<json or short text>"
#
# We give a tiny demo so the 270M model copies the pattern.
system_prompt = (
    "Ты — ассистент для RAG. Отвечай ТОЛЬКО по контексту инструментов. "
    "Если ответа нет — напиши ровно: 'Не знаю.' Коротко (≤ 8 предложений). "
    "Не добавляй преамбул и не описывай свои правила.\n\n"
    "ОБЯЗАТЕЛЬНО: сначала вызови инструмент search_kb. НЕ ДАВАЙ Final Answer, "
    "пока не получишь результат инструмента.\n\n"
    "Формат, которому НУЖНО строго следовать:\n"
    "Thought: <краткая причина вызова инструмента>\n"
    "Action: search_kb\n"
    "Action Input: \"<краткий запрос для инструмента>\"\n"
    "# (после наблюдения от инструмента на следующем шаге)\n"
    "Thought: <краткая проверка что контекст получен>\n"
    "Final Answer: <итоговый ответ на русском>\n"
    "Источник: <URL>  # если инструмент вернул URL\n\n"
    "Пример только первого шага:\n"
    "User: Кто такие key users?\n"
    "Assistant:\n"
    "Thought: нужно получить определение из базы знаний\n"
    "Action: search_kb\n"
    "Action Input: \"key users определение\"\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("messages"),     # Required: the agent feeds conversation here
])

# =========================
# 3) Build the agent (no AgentExecutor)
# =========================
memory = MemorySaver()
app = create_react_agent(
    model=llm,
    tools=[search_kb],       # only one tool => simpler choice for 270M
    prompt=prompt,
    name="assistant_sa",
    checkpointer=memory,
    debug=False,             # set True to see tool call traces
    # Some versions expose max_iterations; if so, 2–3 is ideal:
    # max_iterations=3,
)

# =========================
# 4) Run helper (pure agent; no forced tool calls)
# =========================
def run_query(query: str, thread_id: str = "user-111"):
    config = {"configurable": {"thread_id": thread_id}}
    state = app.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    # The state holds the full message list (user -> assistant/tool turns)
    msgs = state["messages"]
    final = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), msgs[-1])

    print(f"\nQ: {query}\n")
    print("Answer:\n", final.content)
    return final.content

# =========================
# 5) Examples
# =========================
if __name__ == "__main__":
    run_query("Кто такие key users и как к ним обратиться?", thread_id="svc-111")
    run_query("Что такое EL и как он влияет на рассмотрение заявки?", thread_id="svc-111")