from mindware.components.config_space.cs_builder import get_hpo_cs
from mindware.third_party.opro.utils import _build_dataset_summary_text_onehot, _constraints_text_for_cs


origin_to_content = {
    'arxiv': 'arXiv papers',
    'kaggle': 'kaggle notebooks',
    'google': 'google search content'
}


def origin_to_keywords(origin, model, task_type=None):
    if origin == 'arxiv':
        if model is None:
            return 'machine learning, model and hyperparameter'
        else:
            return f'{model}, hyperparameter'
    elif origin == 'kaggle':
        assert task_type is not None
        if model is None:
            return f'machine learning, {task_type}'
        else:
            return f'{model}, {task_type}'
    elif origin == 'google':
        if model is None:
            return "Best practices to select model and set hyperparameters in machine learning."
        else:
            return f"Best practices to set hyperparameters for {model}."
    else:
        raise ValueError



def get_summarize_prompt(origin, documents, user_requirement_summary, model=None, task_type=None):


    keywords = origin_to_keywords(origin, model, task_type)

    if model is None:
        prompt = f"""

**OVERALL MISSION: Automated Machine Learning (AutoML)**
    Summarize insightful knowledge. I will use them to address the user's requirements in machine learning tasks.

**1. Directive:**
    Please extract and summarize the following group of {origin_to_content[origin]} collected from {origin} into insightful knowledge. The knowledge should should focus on the algorithm priority on the dataset. 
1) Make sure that all recommended algorithms **are within the Available Algorithms provided in the User's Requirements**. 2) You should also assign a priority score in the range of [0, 1] to each algorithm based on the insights from the sources.

**2. User's Requirements:**
    {user_requirement_summary}

**3. Available knowledge Sources:**
    I searched the {origin} to find state-of-the-art solutions using the keywords: {keywords}. Here is the result:
    =====================
    {documents}
    =====================

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. You can refer to a few pieces of knowledge from the sources that can help identify the algorithm priority.)

**Algorithm Order and Priority Score:**
[["algorithm_1", priority_score_1], ["algorithm_2, priority_score_2], ...]

        """
    else:
        prompt = f"""

**OVERALL MISSION: Automated Machine Learning (AutoML)**
    Summarize insightful knowledge. I will use them to address the user's requirements in machine learning tasks.

**1. Directive:**
    Please extract and summarize the following group of {origin_to_content[origin]} collected from {origin} into insightful knowledge. The knowledge should should focus on recommend a best hyperparameter setting for {model} and find hyperparameters that have strong interactions and should be tuned simultaneously..
1) The recommended hyperparameter settings **should be within the Available Hyperparameters range provided in the User's Requirements**. 2) You can provide up to 3 **non-overlapping** groups of hyperparameters to be tuned together.


**2. User's Requirements:**
    {user_requirement_summary}

**3. Available knowledge Sources:**
I searched the {origin} to find state-of-the-art solutions using the keywords: {keywords}. Here is the result:
    =====================
    {documents}
    =====================

**4. Required Output Format:**
You must provide your reasoning and your final configuration in this format:

**Thought:**
(Explain your rationale. You can refer to a few pieces of knowledge from the sources that can help identify the hyperparameter settings or interactions.)

**One Recommended Hyperparameter Setting:**
{{
    "hyperparameter_1": value_1,
    "hyperparameter_2": value_2,
    ...
}}

**Hyperparameters that should be tuned together:**
[["hyperparameter_A", "hyperparameter_B", ...], ["hyperparameter_X", "hyperparameter_Y", ...], ...]
        """

    return prompt


def get_summarize_system_prompt(origin, model=None):

    if model is None:
        prompt = f"You are tasked to summarize the contents from the {origin_to_content[origin]} with the goal to provide insightful results in addressing user's requirements. Please pay attention to the state-of-the-art models and their hyperparameters."
    else:
        prompt = f"You are tasked to summarize the contents from the {origin_to_content[origin]} with the goal to provide insightful results in addressing user's requirements. Please pay attention to the hyperparameters setting of {model}."

    return prompt


def get_user_requirement_summary(dataset, dataset_profile, task_type, model=['adaboost', 'extra trees', 'gradient boosting', 'lightgbm', 'random forest', 'xgboost', 'k nearest neighbours'], **cs_args):

    dataset_summary = _build_dataset_summary_text_onehot({'dataset_profile': dataset_profile},model)
    if isinstance(model, list):
        user_requirement_summary = f"""
**USER OVERALL MISSION:**
Optimize a machine learning pipeline througe identifying the best algorithms on the {dataset} dataset.
- **Dataset & Metric Summary:**
{dataset_summary}
- **Available Algorithms:**
The allowable algorithms for the ML task are: {model}.
"""
    else:

        cs = get_hpo_cs(estimator_id=model, task_type=task_type, **cs_args)
        cs_text = _constraints_text_for_cs(cs)
        user_requirement_summary = f"""
**USER OVERALL MISSION:**
Optimize a {model} model through identifying the best hyperparameter settings on the {dataset} dataset.
- **Dataset & Metric Summary:**
{dataset_summary}
- **Available Hyperparameters:**
The allowable hyperparameter for algorithm {model} is:
{cs_text}
"""

    return user_requirement_summary


# # Source: arXiv Papers
#     {arxiv_summary}
# =====================

def get_final_summarize_prompt(search_summary, arxiv_summary, kaggle_summary, historical_summary, user_requirement_summary, model=None):

#    if model is None:
#         output_format_block = """```json
#     {
#     "exploration_algorithms": ["algorithm_1", "algorithm_2", "..."],
#     "exploitation_algorithms": ["algorithm_A", "algorithm_B", "..."]
#     }
# ```"""

#         summary_prompt = f"""
# **OVERALL MISSION: Automated Machine Learning (AutoML)**
#     Summarize insightful knowledge. I will use them to address the user's requirements in machine learning tasks.

# **1. Directive:**
#     Please extract and summarize the following group of contents collected from different online sources into insightful knowledge. Your goal is to categorize available algorithms into two list: `Exploration` and `Exploitation`.

# **2. User's Requirements:**
#     {user_requirement_summary}

# **3. Guiding Principles for Algorithm Categorization:**
# Use these principles to strategically categorize **each algorithm**.
# - **Exploitation Candidates**: Identify **a small, elite group of algorithms** to focus on. A good candidate demonstrates a **Consistent High Performance** across multiple sources.
# - **Exploration Candidates**: Include algorithms that show **Potential** but lack consistent top-tier performance. These algorithms may have **Niche Strengths** in specific scenarios or datasets.
# - Ensure every algorithm from the 'Available Algorithms' is placed in at least one of the updated lists.
# - If inconsistent recommendations are found among different sources, solve it according to the priority order of: Historical Tuning Experience > Kaggle Hub > Google Web Search.

# **4. Available knowledge Sources:**
# # Source: Historical Tuning Experience
#     {historical_summary}
# # =====================

# # Source: Google Web Search
#     {search_summary}
# # =====================

# # Source: Kaggle Hub
#     {kaggle_summary}
# =====================

# **5. Required Output Format:**
# You must provide your rationale and your final knowledge in the following format,

# **Thought:**
# (Explain your rationale. You can refer to a few pieces of knowledge from the sources that can help identify the algorithm priority and how you resolve inconsistent recommendations.)

# **Algorithm Category:**
# (Provide your final knowledge in a single json block. Do not add any text before or after it)
# {output_format_block}

# """
    if model is None:

        output_format_block = """```json
[["algorithm_1", priority_score_1], ["algorithm_2, priority_score_2], ...]
```"""
        summary_prompt = f"""
**OVERALL MISSION: Automated Machine Learning (AutoML)**
    Summarize insightful knowledge. I will use them to address the user's requirements in machine learning tasks.

**1. Directive:**
    Please extract and summarize the following group of contents collected from different online sources into insightful knowledge. The knowledge should should focus on the algorithm priority on the dataset. 
1) Make sure that all recommended algorithms **are within the Available Algorithms provided in the User's Requirements**. 
2) You should also assign a priority score in the range of [0, 1] to each algorithm based on the insights from the sources.
3) If inconsistent recommendations are found among different sources, solve it according to the priority order of: Kaggle Hub > Google Web Search.

**2. User's Requirements:**
    {user_requirement_summary}
**3. Available knowledge Sources:**
# Source: Google Web Search
    {search_summary}
=====================
# Source: Kaggle Hub
    {kaggle_summary}
=====================

**4. Required Output Format:**
You must provide your rationale and your final knowledge in the following format,
**Thought:**
(Explain your rationale. You can refer to a few pieces of knowledge from the sources that can help identify the algorithm priority and how you resolve inconsistent recommendations.)
**Algorithm Order and Priority Score:**
(Provide your final knowledge in a single json block. Do not add any text before or after it)
{output_format_block}
"""
    else:


        summary_prompt = f"""
**OVERALL MISSION: Automated Machine Learning (AutoML)**

**1. Directive:**
    Please recommend some best hyperparameter setting for {model}.
1) The recommended hyperparameter settings **should be within the Available Hyperparameters provided in the User's Requirements**. 
2) The recommended hyperparameter settings should be different from each other and cover diverse areas of the hyperparameter space.
3) You shold take into account the User's Requirements.

**2. User's Requirements:**
    {user_requirement_summary}

**3. Required Output Format:**
You must provide your rationale and your final configuration in the following format:

**Thought:**
(Explain your rationale.)

**Three Recommended Hyperparameter Setting:**
(Recommended Three hyperparameter setting in a single json block. Do not add any text before or after it)
```json
[
    {{
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }},
    {{
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }},
    {{
        "hyperparameter_1": value_1,
        "hyperparameter_2": value_2,
        ...
    }}
]
```

"""

#         summary_prompt = f"""
# **OVERALL MISSION: Automated Machine Learning (AutoML)**
#     Summarize insightful knowledge. I will use them to address the user's requirements in machine learning tasks.

# **1. Directive:**
#     Please extract and summarize the following group of contents collected from different online sources into insightful knowledge. 
#     The knowledge should should focus on recommend some best hyperparameter setting for {model} and find hyperparameters that have strong interactions and should be tuned simultaneously..
# 1) The recommended hyperparameter settings **should be within the Available Hyperparameters provided in the User's Requirements**. 
# 2) The recommended hyperparameter settings should be different from each other.
# 3) You can provide up to **2 non-overlapping** groups of hyperparameters to be tuned together. 
# 4) If inconsistent recommendations are found among different sources, solve it according to the priority order of: Kaggle Hub > Google Web Search.

# **2. User's Requirements:**
#     {user_requirement_summary}

# **3. Available knowledge Sources:**
# # Source: Google Web Search
#     {search_summary}
# =====================

# # Source: Kaggle Hub
#     {kaggle_summary}
# =====================

# **4. Required Output Format:**
# You must provide your rationale and your final configuration in the following format:

# **Thought:**
# (Explain your rationale. You can refer to a few pieces of knowledge from the sources that can help identify the hyperparameter settings or interactions and how you resolve inconsistent recommendations.)

# **Two Recommended Hyperparameter Setting:**
# (Recommended two hyperparameter setting in a single json block. Do not add any text before or after it)
# ```json
# [
#     {{
#         "hyperparameter_1": value_1,
#         "hyperparameter_2": value_2,
#         ...
#     }},
#     {{
#         "hyperparameter_1": value_1,
#         "hyperparameter_2": value_2,
#         ...
#     }}
# ]
# ```

# **Hyperparameters that should be tuned together:**
# (Provide in a single json block. Do not add any text before or after it)
# ```json
# [["hyperparameter_A", "hyperparameter_B", ...], ["hyperparameter_X", "hyperparameter_Y", ...], ...]
# ```

# """

    return summary_prompt