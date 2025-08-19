import os
#os.environ["TRANSFORMERS_VERBOSITY"] = "info"

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


from pprint import pprint

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from retrievers.retriever import get_retriever_object_faiss, get_retriever_object_teamly

# 1) Retriever (consider k=3–5 and MMR in your retriever)
retriever = get_retriever_object_teamly()

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
    temperature=0.0,
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

hf_llm = HuggingFacePipeline(pipeline=pipe)
llm = ChatHuggingFace(llm=hf_llm)

# 3) Prompt — multilingual, concise, grounded, with proper newline
system_prompt = (
    "You are a helpful assistant for retrieval-augmented QA.\n"
    "Use ONLY the given context to answer the question. If the answer is not in the context, say you don't know.\n"
    "Keep the answer concise (<= 8 sentences). If sources (URLs) are shown in context, include the most relevant one.\n"
    "If available return FIRST definition, and only then list.\n"
    "Use Russian language to form your answer.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Ответь на вопрос пользователя на русском языке: {input}"),
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
query = "Кто такие key users?"
result = chain.invoke({"input": query})

# Depending on your LangChain version, output key can be 'answer' or 'output_text'
answer_text = result.get("answer") or result.get("output_text")
pprint(answer_text)

# If you enabled return_source_documents in your retriever/chain, print sources:
src_docs = result.get("context") or result.get("source_documents")
if src_docs:
    urls = []
    for d in src_docs:
        meta = getattr(d, "metadata", {}) or {}
        # Common metadata keys people use: 'source', 'url', 'link', 'path'
        for key in ("url", "source", "link", "path"):
            if key in meta and isinstance(meta[key], str):
                urls.append(meta[key])
                break
    if urls:
        print("\nSources:")
        for u in dict.fromkeys(urls):  # de-duplicate, keep order
            print("-", u)