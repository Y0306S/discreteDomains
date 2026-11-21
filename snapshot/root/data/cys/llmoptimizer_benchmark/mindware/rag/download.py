from mindware.rag.utils import get_kaggle, search_web
from mindware.modules.configs import CONFIGS, AVAILABLE_LLMs
import json
import math

import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

def download_kaggle(model, task_type, top_k=1000, save_path='kaggle/'):

    save_path = os.path.join(CONFIGS.RAG_DATA_DIR, save_path)

    kaggle_api = get_kaggle()


    if model == 'machine learning':
        save_path = os.path.join(save_path, task_type, 'model.json')
    else:
        save_path = os.path.join(save_path, task_type, f'{model}.json')

    search_key = f"{model.replace('_', ' ')}, {task_type}"
    print(f"kaggle search key: {search_key}")

    keep_keys = ["id", "ref", "title", "slug", "categoryIds"]

    documents = []
    count = 0
    page_num = math.ceil(top_k / 50) + 1
    for page in range(page_num):
        notebooks = kaggle_api.kernels_list_with_http_info(
            search=search_key,
            sort_by="relevance",
            language="Python",
            page_size=50,
            page = page + 1
        )[0]
        for notebook in notebooks:
            try:
                notebook = kaggle_api.kernel_pull(*notebook["ref"].split("/"))
                if type(notebook) == str:
                    pass
                else:
                    cells = json.loads(notebook["blob"]["source"])["cells"]
                    page_content = "".join(
                        [
                            (
                                cell["source"]
                                if cell["cell_type"] == "markdown"
                                else f"\n```python\n{cell['source']}\n```"
                            )
                            for cell in cells
                        ]
                    )
                    documents.append(
                        {"page_content": page_content, "metadata": {k: notebook["metadata"][k] for k in keep_keys}}
                    )
            except:
                print("Error!")
                continue

            print(count+1)
            if count % 50 == 0 or count == top_k - 1:

                with open(save_path, 'w') as f:
                    json.dump(documents, f, indent=4)
            
            if count == top_k - 1:
                break

            count += 1

    with open(save_path, 'w') as f:
        json.dump(documents, f, indent=4)

            
import arxivloader
from langchain_community.document_loaders import AsyncHtmlLoader, PDFMinerLoader

def download_arxiv(
    task_kw, domain_kw, top_k=100, save_path='arxiv/'
):

    save_path = os.path.join(CONFIGS.RAG_DATA_DIR, save_path)

    if task_kw == 'machine learning':
        save_path = os.path.join(save_path, 'model.json')
    else:
        save_path = os.path.join(save_path, f'{task_kw}.json')

    task_kw = task_kw.replace('_', ' ')
    query = f'search_query=all:"{task_kw}" AND all:"{domain_kw}" AND (cat:cs.AI OR cat:cs.CV OR cat:cs.LG OR cat:cs.DB)'
    df = arxivloader.load(query, num=top_k, sortBy="relevance", verbosity=0)
    arxiv_links = [links.split(";")[-1].strip() for links in df["links"].tolist()]
    # get paper's pdf
    documents = []
    count = 0
    for link in arxiv_links:
        try:
            documents.append(PDFMinerLoader(link).load()[0].to_json()['kwargs'])
        except Exception as e:
            print('cannot load', link, 'with error:', e)            


        if count % 10 == 0 or count == len(arxiv_links) - 1:
            print(count+1)

            with open(save_path, 'w') as f:
                json.dump(documents, f, indent=4)

        count += 1

    with open(save_path, 'w') as f:
        json.dump(documents, f, indent=4)


def download_google(llm='gpt-4o', user_requirement_summary='Combined Algorithm Selection and Hyperparameter Optimization'
                    , top_k=100, save_path='google/', model=None):

    save_path = os.path.join(CONFIGS.RAG_DATA_DIR, save_path)

    from openai import OpenAI
    from langchain_community.document_transformers import BeautifulSoupTransformer
    from validators import url
    # llm_model = AVAILABLE_LLMs[llm]["model"]
    
    # client = OpenAI(
    #     base_url=AVAILABLE_LLMs[llm]["base_url"],
    #     api_key=AVAILABLE_LLMs[llm]["api_key"],
    # )
    # query_profile = f"You are tasked with generating web search queries based on the given machine learning problem. Give me a specific query for google search focusing on the downstream tasks. Please give me a single sentence within 10 words. The answer should not exceed 10 words."

    # # genearte query
    # while True:
    #     try:
    #         response = client.chat.completions.create(
    #             model=llm_model,
    #             messages=[
    #                 {"role": "system", "content": query_profile},
    #                 {"role": "user", "content": user_requirement_summary},
    #             ],
    #             temperature=0.1,
    #         )
    #         break
    #     except Exception as e:
    #         print('system', e)
    #         continue              

    # search_query = (
    #     response.choices[0].message.content.strip().replace('"', "").replace(".", "")
    # )


    if model == 'machine learning':
        save_path = os.path.join(save_path, 'model.json')
    else:
        save_path = os.path.join(save_path, f'{model}.json')

    if model is None:
        search_query = "Best practices to select model and set hyperparameters in machine learning."
    else:
        search_query = f"Best practices to set hyperparameters for {model.replace('_', ' ')}."

    print(
        "manager", f"I am searching Google using query 🔍: {search_query}."
    )

    documents = []

    url_set = set()
    count = 1
    while count < top_k:
        search_results = search_web(search_query, start=count)
        DOMAIN_BLOCKLIST = [
            "youtube.com",
            "twitter.com",
            "x.com",
            "hindawi.com",
            "ejournal.ittelkom-pwt.ac.id",
        ]

        count += len(search_results)
        search_results = [
            result
            for result in search_results
            if not any(domain in result["link"] for domain in DOMAIN_BLOCKLIST)
        ]
        # print('Search results:', search_results)
        urls = [link["link"] for link in search_results if url(link["link"])]

        for link in urls:
            if '.pdf' in link: continue
            if link in url_set: 
                print("Already")
                continue

            url_set.add(link)
            try:
                loader = AsyncHtmlLoader([link])
                html = loader.load()
                print("success")
            except:
                print("failed")
                continue

            bs_transformer = BeautifulSoupTransformer()
            html_docs = bs_transformer.transform_documents(
                html, tags_to_extract=["p", "li", "div", "span", "table"]
            )

            for link in urls:
                if (
                    "arxiv.org/pdf" in link
                    or "/pdf?id=" in link
                    or "&name=pdf" in link
                ):
                    html_docs += PDFMinerLoader(link).load()


            for html in html_docs:
                documents.append(html.to_json()['kwargs'])

        with open(save_path, 'w') as f:
            json.dump(documents, f, indent=4)

        print(f"=================== Count {count} ==================")



if __name__ == '__main__':

    # for task_type in ['classification', 'regression']:
    #     download_kaggle(model='machine learning', task_type=task_type, top_k=1000, save_path='_data/kaggle/')
    for task_type in ['classification', 'regression']:
        for model in ['random_forest', 'extra_trees', 'gradient_boosting', 'k_nearest_neighbors']:  # 'adaboost', 'lightgbm', 'xgboost', 

            download_kaggle(model=model, task_type=task_type, top_k=100, save_path='kaggle/')
    
    # download_arxiv('machine learning', 'model and hyperparameter', top_k=100, save_path='_data/arxiv/paper.json')
    for model in ['adaboost', 'extra_trees', 'gradient_boosting', 'random_forest', 'lightgbm', 'xgboost', 'k_nearest_neighbors']:
        download_arxiv(task_kw=model, domain_kw='hyperparameter', top_k=10, save_path='arxiv/')

    # download_google(top_k=100)

    for model in ['adaboost', 'extra_trees', 'gradient_boosting', 'random_forest', 'lightgbm', 'xgboost', 'k_nearest_neighbors']:
        download_google(model=model, top_k=20, save_path='google/')

# machine learning, classification, 1000
# machine learning, regression, 1000

# machine learning, model and hyperparameter, 100

# Best practices to select model and set hyperparameters in machine learning, 100