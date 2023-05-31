from typing import Dict, Callable, Optional, List, Any, Union

import numpy as np


def percent_str(num: int, num_total: int, decimals: int = 1) -> str:
    return f'{round(100 * num / num_total, decimals)}%'


def flatten_dictionary(dictionary: Dict, agg_label_fn: Optional[Callable[[List[str]], str]] = None) -> Dict:

    def join_with_whitespace(entries: List[str]) -> str:
        return ' '.join(entries)

    if agg_label_fn is None:
        agg_label_fn = join_with_whitespace

    def recursive_flatten(current_element: Union[Dict, Any], current_keys: List[str], result: Dict) -> Dict:
        if type(current_element) is not dict:
            result[agg_label_fn(current_keys)] = current_element
        else:
            for key in current_element:
                recursive_flatten(current_element[key], current_keys + [key], result)
        return result

    return recursive_flatten(dictionary, [], dict())


def agg_dictionaries(
        dictionaries: List[Dict],
        agg_label_fn: Optional[Callable[[List[str]], str]] = None
) -> Dict[str, List]:
    flattened_dictionaries: List[Dict] = list(
        map(lambda x: flatten_dictionary(x, agg_label_fn=agg_label_fn), dictionaries)
    )

    return {
        key: [flattened_dictionaries[i][key] for i in range(len(flattened_dictionaries))]
        for key in flattened_dictionaries[0].keys()
    }


def pretty_print_averaged_results(classification_reports: List[Dict], decimals: int = 3) -> Dict:
    agg_classification_reports: Dict[str, List] = agg_dictionaries(classification_reports)
    print('Average results over', len(classification_reports), 'reports.')
    for key in agg_classification_reports:
        values: List[float] = agg_classification_reports[key]
        min_value: float = round(np.min(values), decimals)
        max_value: float = round(np.max(values), decimals)
        avg_value: float = round(float(np.mean(values)), decimals)
        std_value: float = round(float(np.std(values)), decimals)
        all_values = list(map(lambda x: round(x, 3), values))
        print(f'{key}: {avg_value} (+/- {std_value}); min: {min_value}, max: {max_value}; all={all_values}')

    return agg_classification_reports
