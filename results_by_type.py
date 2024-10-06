import json
import pickle
import numpy as np


def get_saved_results(results_path):
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    return results


def get_prompts_by_types(path):
    with open(path, "r") as file:
        prompts = json.load(file)
    return prompts


def get_results_by_type(results):
    animals_results = {
        k: v for k, v in results.items() if k in prompts_by_types["animals"]
    }
    animals_objects_results = {
        k: v for k, v in results.items() if k in prompts_by_types["animals_objects"]
    }
    objects_results = {
        k: v for k, v in results.items() if k in prompts_by_types["objects"]
    }
    return animals_results, animals_objects_results, objects_results


def aggregate_by_full_text(d):
    """Aggregate results for the full text similarity for each prompt."""
    full_text_res = [v["full_text"] for v in d.values()]
    full_text_res = [x for x in full_text_res if len(x) == 10]
    full_text_res = np.array(full_text_res).flatten()
    return np.average(full_text_res)


def aggregate_by_min_half(d):
    """Aggregate results for the minimum similarity score for each prompt."""
    min_per_half_res = [
        [min(a, b) for a, b in zip(d[prompt]["first_half"], d[prompt]["second_half"])]
        for prompt in d
    ]
    min_per_half_res = [x for x in min_per_half_res if len(x) == 10]
    min_per_half_res = np.array(min_per_half_res).flatten()
    return np.average(min_per_half_res)


def aggregate_text_similarities(result_dict):
    all_averages = [
        result_dict[prompt]["text_similarities"] for prompt in result_dict.keys()
    ]
    all_averages = [x for x in all_averages if len(x) == 10]
    all_averages = np.array(all_averages).flatten()
    total_average = np.average(all_averages)
    total_std = np.std(all_averages)
    return total_average, total_std


def get_aggregated_image_text_results(results):
    animals_results, animals_objects_results, objects_results = get_results_by_type(
        results
    )
    animals_full_text = aggregate_by_full_text(animals_results)
    animals_objects_full_text = aggregate_by_full_text(animals_objects_results)
    objects_full_text = aggregate_by_full_text(objects_results)
    animals_min_half = aggregate_by_min_half(animals_results)
    animals_objects_min_half = aggregate_by_min_half(animals_objects_results)
    objects_min_half = aggregate_by_min_half(objects_results)
    return {
        "animals_full_text": animals_full_text,
        "animals_objects_full_text": animals_objects_full_text,
        "objects_full_text": objects_full_text,
        "animals_min_half": animals_min_half,
        "animals_objects_min_half": animals_objects_min_half,
        "objects_min_half": objects_min_half,
    }


def get_aggregated_text_text_results(results):
    animals_results, animals_objects_results, objects_results = get_results_by_type(
        results
    )
    animals_avg, animals_std = aggregate_text_similarities(animals_results)
    animals_objects_avg, animals_objects_std = aggregate_text_similarities(
        animals_objects_results
    )
    objects_avg, objects_std = aggregate_text_similarities(objects_results)
    return {
        "animals_avg": animals_avg,
        # "animals_std": animals_std,
        "animals_objects_avg": animals_objects_avg,
        # "animals_objects_std": animals_objects_std,
        "objects_avg": objects_avg,
        # "objects_std": objects_std,
    }


image_text_results = get_saved_results("image_text_results.pkl")
text_text_results = get_saved_results("text_text_results.pkl")
prompts_by_types = get_prompts_by_types("ane_test_prompts.txt")

aggregated_image_text_results = get_aggregated_image_text_results(image_text_results)
aggregated_text_text_results = get_aggregated_text_text_results(text_text_results)

print("Aggregated image-text results:")
print(aggregated_image_text_results)
print("Aggregated text-text results:")
print(aggregated_text_text_results)
