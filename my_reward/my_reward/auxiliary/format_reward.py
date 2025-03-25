from itertools import islice, zip_longest
from collections import defaultdict
import re

def score_think_pattern(s: str, not_need_think_at_start: bool = False, not_need_answer_tag: bool = False, overlong: bool = False):
    s = s.strip()
    # 1. Whether it starts with "<think>"
    if not not_need_think_at_start:
        if not s.startswith("<think>"):
            return 0.0
    # 2. Whether it contains "<think>...</think>" only once and does not contain <answer>...</answer> inside
    # 2.1 Whether there are multiple "<think>"
    if (not_need_think_at_start and s.count("<think>") > 0) or (not not_need_think_at_start and s.count("<think>") > 1):
        return 0.0
    # 2.2 Whether there are multiple "</think>"
    if s.count("</think>") > 1:
        return 0.0
    think_end = s.find("</think>")
    # 2.3 Whether it contains "</think>"
    if think_end == -1:
        if "<answer>" in s or "</answer>" in s:
            return 0.0
        # over max length
        elif overlong:
            return 0.4
        else:
            return 0.0
    think_content = s[len("<think>"):think_end] if not not_need_think_at_start else s[:think_end]
    # 2.4 Must not contain <answer> </answer>
    if "<answer>" in think_content or "</answer>" in think_content:
        return 0.0
    # 3. Whether it contains <answer>...</answer> only once and must be after "<think>...</think>"
    answer_content = s[think_end + len("</think>"):]
    answer_content = answer_content.strip()
    if len(answer_content) == 0:
        return 0.0
    if not_need_answer_tag:
        return 1.0
    # 3.1 Whether it contains <answer>
    answer_start = answer_content.find("<answer>")
    if answer_start == -1:
        return 0.0
    # 3.2 Whether there is extra content before <answer>
    if answer_start != 0:
        return 0.0
    # 3.3 Whether there is only one <answer>
    if answer_content.count("<answer>") > 1:
        return 0.0
    # 3.4 Whether it contains </answer>
    # over max length
    if answer_content.count("</answer>") == 0:
        if overlong:
            return 0.5
        else:
            return 0.0
    # 3.5 Whether there is extra content after </answer>
    answer_end = answer_content.find("</answer>")
    answer_content_last = answer_content[answer_end + len("</answer>"):]
    for x in ["</s>", "<|im_end|>", "<|endoftext|>", "<|end_of_sentence|>"]:
        answer_content_last = answer_content_last.replace(x, "")
    if len(answer_content_last.strip()) != 0:
        return 0.0
    return 1.0

def endswith_think(s: str):
    for x in ["</s>", "<|im_end|>", "<|endoftext|>", "<|end_of_sentence|>"]:
        s = s.replace(x, "")
    s = s.strip()
    return s.endswith("<think>")

def get_think_and_answer(s: str):
    think_content = ""
    answer_content = ""
    if s.startswith("<think>"):
        s = s[len("<think>"):]
    if "</think>" in s:
        think_content = s[:s.find("</think>")]
        answer_content = s[s.find("</think>") + len("</think>"):]
    else:
        answer_content = s
    if "<answer>" in answer_content:
        answer_content = answer_content[answer_content.find("<answer>") + len("<answer>"):]
    if "</answer>" in answer_content:
        answer_content = answer_content[:answer_content.find("</answer>")]
    think_content = think_content.strip()
    answer_content = answer_content.strip()
    return think_content, answer_content

def score_repeatness(s: str):
    """
    Calculate the repeatness score. The higher the score, the higher the proportion of repeated substrings.
    Score range: [0, 1]
    """

    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        # suffixArray: Constructs the suffix array using the doubling algorithm
        line = ranks(s)
        # print(line)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            # print(line, k)
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        # The sa array represents the starting position of the suffix with rank i.
        # The ans array, corresponding to the rk array, represents the rank of the suffix starting at position i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        # suffixArr denotes sa
        # inv_suff denotes rk
        n, ans, k = len(arr), [0] * len(arr), 0
        # ans corresponds to LCP(i, i+1)
        # LCP(i, i+1) represents the longest common prefix between the suffixes at ranks i and i+1
        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue
            # j, the starting position of the suffix with the next rank to the current position i
            # Compare two adjacent ranked suffixes to find the longest common prefix
            j = suffixArr[inv_suff[i] + 1]
            # Count the length of the longest common prefix
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1
            # Record the result for the current rank at position i
            ans[inv_suff[i]] = k
            # Starting from the next position, it is known that the position is one step ahead of the previous prefix position
            # If the previous longest common prefix is k, then the next longest common prefix is at least k-1, unless it is the last position
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))
    # The sum of the longest common prefixes for all suffixes divided by the sum of the lengths of all suffixes
    return cnt * 2 / (n * (n + 1))

def score_reflection_pattern(s: str):
    """
    The number of reflective words included in the response, a non-normalized value.
    """
    # TODO: may need to add more pattern
    reflection_pattern_words = [
        r"wait,",
        r"recheck[,\s]",
        r"retry",
        r"alternatively,",
        r"however,",
        r"therefore,",
        r"given that",

    ]
    s = s.lower()
    res = defaultdict(int)
    for word in reflection_pattern_words:
        # can only be followed by a comma or a space
        res[word] = len(re.findall(word, s))
    return sum(res.values())


if __name__ == "__main__":
    response = "对于痰量增多、脓血痰且有臭味的支气管炎患者，应景授予氨基糖苷类等抗菌药。可在当前��阿奇霉素、头孢呋辛和喹诺酮类抗菌药的基础上，补充加用喹诺酮类抗菌药治疗。</think><answer>喹诺酮类抗菌药治疗</answer>\n\n<|endoftext|>"
    print(
        score_think_pattern(
            response, 
            not_need_think_at_start=True, 
            not_need_answer_tag=False
        )
    )