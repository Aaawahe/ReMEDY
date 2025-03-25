R1_ORIGIN_PROMPT_ADD_LANGUAGE = """
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind (invisible to the User) and then provides the User with the answer. \
The step-by-step reasoning process is enclosed within <think> </think> and followed by the wrap-up answer, i.e., <think> reasoning process here </think> answer here. \
Think and answer with the same language as the question.
User: {prompt}
Assistant: <think>
""".strip()