import os
from functools import partial
from collections import defaultdict
from verl.workers.reward_manager.http_utils import do_parallel_request


def compute_score_by_actor(params, actor_list, data_source_list, prompt_str_list, response_str_list, ground_truth_list, extra_info_list, finish_reason_list):
    import my_reward
    # Group by actor
    actor_order_list = list(enumerate(actor_list))
    actor_group = defaultdict(list)
    for i, actor in actor_order_list:
        actor_group[actor].append(i)
    # Compute separately by actor
    results = []
    for actor, index_list in actor_group.items():
        actor = eval(f"my_reward.contrib.{actor}")
        actor_results = actor.batch_compute_score(
            params=params,
            data_source_list=[data_source_list[i] for i in index_list],
            prompt_str_list=[prompt_str_list[i] for i in index_list],
            response_str_list=[response_str_list[i] for i in index_list],
            ground_truth_list=[ground_truth_list[i] for i in index_list],
            extra_info_list=[extra_info_list[i] for i in index_list],
            finish_reason_list=[finish_reason_list[i] for i in index_list],
        )
        results.extend([(index_list[i], actor_results[i]) for i in range(len(index_list))])
    
    results = sorted(results, key=lambda x: x[0])
    results = [x[1] for x in results]
    return results


def get_compute_score_func(config):
    compute_score = None

    # remote model to verify + local actor
    if config.reward_model.get("my_reward_verify_url") is not None:
        import my_reward
        my_reward_verify_url = config.reward_model.my_reward_verify_url
        params = {
            "url": my_reward_verify_url,
            "model": config.reward_model.my_reward_verify_model,
            "key": config.reward_model.my_reward_verify_key,
        }
        params["max_concurrency"] = config.reward_model.get("my_reward_verify_max_concurrency", 8)
        params["max_tokens"] = config.reward_model.get("my_reward_verify_max_tokens", 4096)
        compute_score = partial(compute_score_by_actor, params=params)

    return compute_score