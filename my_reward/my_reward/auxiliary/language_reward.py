import re

def score_language_consistency(s1: str, s2: str, split_char_length: int = 400):
    """
    Language consistency score
    Score range [0, 1]
    s2 is segmented and language is evaluated, with the final score being the proportion of the target language.
    Only distinguishes whether Chinese is included.
    """
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    lang1 = bool(zh_pattern.search(s1))
    
    lang2_list = []
    for i in range(0, len(s2), split_char_length):
        lang2 = bool(zh_pattern.search(s2[i:i+split_char_length]))
        lang2_list.append(lang2)
    
    if len(lang2_list) == 0:
        return 0.0
    
    return lang2_list.count(lang1) * 1.0 / len(lang2_list)
