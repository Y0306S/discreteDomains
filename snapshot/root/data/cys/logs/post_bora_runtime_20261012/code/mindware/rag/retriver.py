# import arxivloader, json, random
import json, random
from pathlib import Path
from mindware.rag.utils import search_web, print_message
from mindware.rag.utils.prompt import get_summarize_prompt, get_summarize_system_prompt, get_final_summarize_prompt
# from mindware.rag.utils.embeddings import chunk_and_retrieve
from mindware.modules.configs import CONFIGS, AVAILABLE_LLMs
from mindware.components.config_space.cs_builder import get_hpo_cs
from mindware.components.utils.constants import CLS_TASKS
from openai import OpenAI
# from validators import url
import time

from langchain.schema import Document


import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

# def retrieve_kaggle(
#     task_type, user_requirement_summary: str, llm_model, client, top_k: int = 10, model=None, data_dir = CONFIGS.RAG_DATA_DIR
# ):

#     task_type = 'classification' if task_type in CLS_TASKS else 'regression'

#     if model is None:
#         data_source = os.path.join(data_dir, 'kaggle/', task_type, 'model.json')
#     else:
#         data_source = os.path.join(data_dir, 'kaggle/', task_type, f'{model}.json')

#     with open(data_source, 'r') as f:
#         data = json.load(f)

#     documents = []
#     for notebook in data:
#         try:

#             documents.append(
#                 Document(**notebook)
#             )
#         except:
#             continue

#     retrieve_chunk = chunk_and_retrieve(
#                 ref_text=user_requirement_summary,
#                 documents=documents,
#                 top_k=top_k,
#                 ranker="bm25",
#             )

#     context = "".join(
#         [
#             d.page_content
#             for d in retrieve_chunk
#         ]
#     )

#     # genearte summmary
#     summary_prompt = get_summarize_prompt(origin='kaggle', documents=context, user_requirement_summary=user_requirement_summary, model=model, task_type=task_type)

#     while True:
#         try:
#             response = client.chat.completions.create(
#                 model=llm_model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": get_summarize_system_prompt(origin='kaggle', model=model),
#                     },
#                     {"role": "user", "content": summary_prompt},
#                 ],
#                 temperature=0.3,
#             )
#             break
#         except Exception as e:
#             print_message('system', e)
#             continue

#     return response.choices[0].message.content.strip()


# def retrieve_arxiv(
#     user_requirement_summary: str, llm_model, client, top_k: int = 10, model=None, data_dir = CONFIGS.RAG_DATA_DIR
# ):

#     if model is None:
#         data_source = os.path.join(data_dir, 'arxiv/', 'paper.json')
#     else:
#         data_source = os.path.join(data_dir, 'arxiv/', f'{model}.json')

#     with open(data_source, 'r') as f:
#         data = json.load(f)

#     documents = []
#     for notebook in data:
#         try:

#             documents.append(
#                 Document(**notebook)
#             )
#         except:
#             continue

#     # generate context from PDF for summary
#     context = "".join(
#         [
#             d.page_content
#             for d in chunk_and_retrieve(
#                 ref_text=user_requirement_summary,
#                 documents=documents,
#                 top_k=top_k,
#                 ranker="bm25",
#             )
#         ]
#     )
#     # genearte summmary
#     summary_prompt = get_summarize_prompt(origin='arxiv', documents=context, user_requirement_summary=user_requirement_summary, model=model)

#     while True:
#         try:
#             response = client.chat.completions.create(
#                 model=llm_model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": get_summarize_system_prompt(origin='arxiv', model=model),
#                     },
#                     {"role": "user", "content": summary_prompt},
#                 ],
#                 temperature=0.3,
#             )
#             break
#         except Exception as e:
#             print_message('system', e)
#             continue

#     return response.choices[0].message.content.strip()


# def retrieve_websearch(user_requirement_summary: str, llm_model, client, top_k: int = 10, model=None, data_dir = CONFIGS.RAG_DATA_DIR):

#     if model is None:
#         data_source = os.path.join(data_dir, 'google/', 'google.json')
#     else:
#         data_source = os.path.join(data_dir, 'google/', f'{model}.json')

#     with open(data_source, 'r') as f:
#         data = json.load(f)

#     documents = []
#     for notebook in data:
#         try:

#             documents.append(
#                 Document(**notebook)
#             )
#         except:
#             continue

#     # generate context from HTML pages for summary
#     context = "".join(
#         [
#             d.page_content
#             for d in chunk_and_retrieve(
#                 ref_text=user_requirement_summary,
#                 documents=documents,
#                 top_k=top_k,
#                 ranker="bm25",
#             )
#         ]
#     )
#     # genearte summmary
#     summary_prompt = get_summarize_prompt(origin='google', documents=context, user_requirement_summary=user_requirement_summary, model=model)

#     while True:
#         try:
#             system_prompt = get_summarize_system_prompt(origin='google', model=model)
#             response = client.chat.completions.create(
#                 model=llm_model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": system_prompt,
#                     },
#                     {"role": "user", "content": summary_prompt},
#                 ],
#                 temperature=0.3,
#             )
#             break
#         except Exception as e:
#             print_message('system', e)
#             continue

#     return response.choices[0].message.content.strip()


def retrieve_knowledge(
    task_type: str, user_requirement_summary: str, llm_model, client, top_k, data_dir = CONFIGS.RAG_DATA_DIR
):

    search_summary = retrieve_websearch(user_requirement_summary, llm_model=llm_model, client=client, top_k=top_k, data_dir=data_dir)
    print(search_summary)
    arxiv_summary = retrieve_arxiv(user_requirement_summary, llm_model=llm_model, client=client, top_k=top_k, data_dir=data_dir)
    print(arxiv_summary)
    kaggle_summary = retrieve_kaggle(task_type, user_requirement_summary, llm_model=llm_model, client=client, top_k=top_k, data_dir=data_dir)
    print(kaggle_summary)

    summary_profile = "You are a senior consultant and a professor in machine learning (ML) and artificial intelligence (AI). You are knowledgable and have a lot of insightful expereinces in ML/AI research."

    summary_prompt = f"""Please extract and summarize the following group of contents collected from different online sources into a chunk of insightful knowledge. I will use them to address the user's requirements in machine learning tasks.
    Make sure that: 1) Please format your answer as a list of suggestions. 2) You should only include specific suggestions about model priority and hyperparameters setting of different models. You should not include other unrelated information like tuning methods.

    # Source: Google Web Search
    {search_summary}
    =====================

    # Source: arXiv Papers
    {arxiv_summary}
    =====================

    # Source: Kaggle Hub
    {kaggle_summary}
    =====================

    The user's requirements are summarized as follows.
    {user_requirement_summary}
    """
    retry = 0
    while retry < 5:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": summary_profile},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:
            print_message('system', e)
            retry += 1
            continue

    return response.choices[0].message.content.strip()



def retrieve_decompose_knowledge(
    task_type, user_requirement_summary_dict: dict, llm_model, client, top_k, include_algorithms, data_dir = './_data'
):

    insights = {}

    for model in include_algorithms:
        start = time.time()
        user_requirement_summary = user_requirement_summary_dict[model]

        # search_summary = retrieve_websearch(user_requirement_summary, llm_model=llm_model, client=client, top_k=top_k, model=model, data_dir=data_dir)
        # arxiv_summary = retrieve_arxiv(user_requirement_summary, llm_model=llm_model, client=client, top_k=top_k, model=model, data_dir=data_dir)
        # kaggle_summary = retrieve_kaggle(task_type, user_requirement_summary, llm_model=llm_model, client=client, model=model, top_k=top_k, data_dir=data_dir)

        summary_profile = "You are a senior consultant and a professor in machine learning (ML) and artificial intelligence (AI). You are knowledgable and have a lot of insightful expereinces in ML/AI research."

        summary_prompt = get_final_summarize_prompt(search_summary=None, arxiv_summary=None, kaggle_summary=None, historical_summary=None, user_requirement_summary=user_requirement_summary, model=model)
        retry = 0
        while retry < 5:
            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": summary_profile},
                        {"role": "user", "content": summary_prompt},
                    ],
                    temperature=0.3,
                )
                break
            except Exception as e:
                print_message('system', e)
                retry += 1
                continue

        insights[model] = response.choices[0].message.content.strip()

        print(f"solve {model}, cost {time.time() - start} s")

    return insights


def retrieve_algorithm_category_knowledge(
    task_type, user_requirement_summary, llm_model, client, top_k, historical_knowledge=None, data_dir = './_data'
):

    start = time.time()

    search_summary = retrieve_websearch(user_requirement_summary, llm_model=llm_model, client=client, top_k=top_k, model=None, data_dir=data_dir)
    # arxiv_summary = retrieve_arxiv(user_requirement_summary, llm_model=llm_model, client=client, top_k=top_k, model=model, data_dir=data_dir)
    kaggle_summary = retrieve_kaggle(task_type, user_requirement_summary, llm_model=llm_model, client=client, model=None, top_k=top_k, data_dir=data_dir)

    summary_profile = "You are a senior consultant and a professor in machine learning (ML) and artificial intelligence (AI). You are knowledgable and have a lot of insightful expereinces in ML/AI research."

    summary_prompt = get_final_summarize_prompt(
        search_summary=search_summary, arxiv_summary=None, kaggle_summary=kaggle_summary, historical_summary=historical_knowledge,
        user_requirement_summary=user_requirement_summary, model=None
    )
    retry = 0
    while retry < 5:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": summary_profile},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
            )
            break
        except Exception as e:
            print_message('system', e)
            retry += 1
            continue

    print(f"solve algorithm categorization, cost {time.time() - start} s")

    return response.choices[0].message.content.strip()