import os
#os.environ["TRANSFORMERS_VERBOSITY"] = "debug"

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

import config

from pprint import pprint

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_gigachat import GigaChat

from retrievers.retriever import get_retriever_object_faiss, get_retriever_object_teamly, get_retriever_object_faiss_chunked

# 1) Retriever (consider k=3–5 and MMR in your retriever)
retriever = get_retriever_object_faiss_chunked()

# 2) Model + tokenizer + safe generation config
model_path = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="cpu",
)

# Make sure generation stops at EOS and padding is defined
end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
eos_id = end_id if isinstance(end_id, int) and end_id >= 0 else tokenizer.eos_token_id
if model.generation_config is not None:
    model.generation_config.eos_token_id = eos_id
    model.generation_config.pad_token_id = eos_id

# Two sensible decoding profiles:
GEN_KWARGS_QA_STRICT = dict(
    max_new_tokens=512,
    do_sample=False,          # greedy for factual QA
    #temperature=0.0,
)
GEN_KWARGS_QA_BALANCED = dict(
    max_new_tokens=512,
    do_sample=True,           # a touch of sampling if you want fuller answers
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.05,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    clean_up_tokenization_spaces=True,
    **GEN_KWARGS_QA_STRICT,
)

hf_llm = HuggingFacePipeline(pipeline=pipe, verbose=True)
llm = ChatHuggingFace(llm=hf_llm)
#llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, frequency_penalty=0)
#"""
llm = GigaChat(
    credentials=config.GIGA_CHAT_AUTH, 
    model="GigaChat-2",
    verify_ssl_certs=False,
    temperature=0,
    frequency_penalty=0,
    scope = config.GIGA_CHAT_SCOPE)
#"""

# 3) Prompt — multilingual, concise, grounded, with proper newline
system_prompt = (
    "You are a helpful assistant for retrieval-augmented QA.\n"
    "Use ONLY the given context to answer the question. If the answer is not in the context, say you don't know.\n"
    "Keep the answer concise (<= 8 sentences). If sources (URLs) are shown in context, include the most relevant one.\n"
    #"If available return FIRST definition, and only then list.\n"
    "Use Russian language to form your answer.\n\n"
    "Context:\n{context}"
)

system_prompt = (
    "Ты — полезный ассистент для retrieval-augmented QA (RAG).\n"
    "Отвечай ТОЛЬКО на основе Контекста.\n"
    "Не используй внешние знания и не додумывай факты.\n"
    "Если ответа в контексте нет, ответь ровно: 'Не знаю.'\n"
    #"Убедись, что ты используешь только ту информацию из Контекста, которая необходима для ответа на вопрос пользователя.\n"
    "Дай ответ, используя Контекст, не более 8 предложений.\n"
    #"Если в контексте есть источники (URL), укажи самый релевантный.\n"
    #"Если доступно, сначала приведи определение, и только затем список.\n"
    "Отвечай на русском языке сухо, соблюдая правила грамматики.\n"
    #"Всегда возвращай Ссылку на статью или файл, если доступна в контексте.\n"
    "Не цитируй и не пересказывай весь контекст, не добавляй вступлений и объяснений правил.\n\n"
    "Ограничение: ответ не должен превышать 512 символов.\n"
)

system_prompt=(
"РОЛЬ: Ассистент для точных ответов на основе Контекста.\n\n"

"ПРАВИЛА:\n"
"1. Используй ТОЛЬКО информацию из Контекста\n"
"2. Используй ВСЮ информацию из Контекста\n"
"3. Если в Контексте ответа нет - отвечай 'Не знаю'\n"
#"4. НИКОГДА не используй информацию, которой нет в Контексте.\n"
"4. Максимум 6-7 предложений\n"
"5. Без цитирования всего контекста\n\n"

"ФОРМАТ ОТВЕТА: Краткий, точный ответ на русском языке.\n\n"

"Контекст: [КОНТЕКСТ]\n"
"Вопрос: [ВОПРОС]\n"
"Ответ:\n"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Верни Контекст."),
        ("ai", "#Контекст:\n{context}#Конец контекста\n\n"),
        ("human", "#Вопрос:\n{input}\n\nОтвечай на русском языке, используя Контекст."),
    ]
)

# 4) Build the combine chain and retrieval chain
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    # Optional: tell the combiner which variable holds docs if you customize
    document_variable_name="context",
)
chain = create_retrieval_chain(retriever, question_answer_chain)

# 5) Ask


queries = [
    "По каким причинам могут вернуть на доработку задачу 'Согласовать СКК'?",
    #"Что такое EL и как он влияет на рассмотрение заявки?",
]

for query in queries:
    result = chain.invoke({"input": query})

    # Depending on your LangChain version, output key can be 'answer' or 'output_text'
    answer_text = result.get("answer") or result.get("output_text")
    print(f"Question: {query}\n\n")
    print("Answer===================>:\n")
    pprint(answer_text)
    #print("<===================\n\n")
    ctx_doc     = result.get("context")[0]
    #pprint(ctx_doc.page_content)
    file    = ctx_doc.metadata["source"]
    print(f"Ссылка на файл: {file}")
    #space_id    = ctx_doc.metadata["space_id"]
    #article_id  = ctx_doc.metadata["article_id"]
    #print(f"Ссылка на статью:https://kb.ileasing.ru/space/{space_id}/article/{article_id}")
    print("<===================\n\n")
    
