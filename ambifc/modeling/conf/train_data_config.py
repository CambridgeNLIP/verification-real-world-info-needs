from typing import Dict


class TrainDataConfig:

    # All samples in the certain subset
    SUBSET_CERTAIN_ONLY: str = 'ambifc-certain'

    # All samples in the certain subset with 5+ annotations.
    SUBSET_CERTAIN_FIVE_PLUS: str = 'ambifc-certain-5+'

    # All samples in the used uncertain subset, i.e. (only) all samples that are uncertain with 5+ annotations.
    SUBSET_UNCERTAIN_ONLY: str = 'ambifc-uncertain-5+'

    # All samples in the  uncertain subset, i.e. (only) all samples that are uncertain with any number of annotations.
    SUBSET_UNCERTAIN_ONLY_ALL: str = 'ambifc-all-uncertain'

    # All samples used in our experiments.
    SUBSET_ALL_AMBIFC: str = 'ambifc'

    # All annotated samples without filtering
    SUBSET_ALL_ANNOTATED: str = 'all-annotated'

    def __init__(self, config: Dict):
        self.config: Dict = config

    def get_data_directory(self) -> str:
        return self.config['directory']

    def is_include_entity_name(self) -> bool:
        return self.config['include_entity_name']

    def is_include_section_header(self) -> bool:
        return self.config['include_section_header']

    def get_ambifc_subset(self) -> str:
        return self.config['ambifc_subset']
