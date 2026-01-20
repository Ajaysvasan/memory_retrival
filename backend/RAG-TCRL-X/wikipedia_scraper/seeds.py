from typing import Dict, List


class WikipediaSeeds:
    """Seed pages organized by topic"""

    SEEDS: Dict[str, List[str]] = {
        "global_mortality": [
            "List of causes of death by rate",
            "Mortality rate",
            "Global health",
            "Epidemiology",
            "Life expectancy",
        ],
        "cardiovascular": [
            "Cardiovascular disease",
            "Myocardial infarction",
            "Stroke",
            "Heart failure",
            "Atherosclerosis",
            "Hypertension",
        ],
        "cancer": [
            "Cancer",
            "Carcinogenesis",
            "Oncology",
            "Lung cancer",
            "Breast cancer",
            "Colorectal cancer",
        ],
        "infectious_disease": [
            "Infectious disease",
            "Tuberculosis",
            "HIV/AIDS",
            "Malaria",
            "Pneumonia",
            "Influenza",
        ],
        "public_health": [
            "Public health",
            "Preventive healthcare",
            "Health policy",
            "Universal health care",
            "World Health Organization",
            "Vaccination",
        ],
        "respiratory": [
            "Respiratory disease",
            "Chronic obstructive pulmonary disease",
            "Asthma",
            "Pulmonology",
            "Respiratory system",
        ],
        "diabetes": [
            "Diabetes",
            "Type 1 diabetes",
            "Type 2 diabetes",
            "Insulin",
            "Diabetic complications",
        ],
        "mental_health": [
            "Mental disorder",
            "Depression (mood)",
            "Anxiety disorder",
            "Schizophrenia",
            "Bipolar disorder",
            "Psychiatry",
        ],
    }

    @classmethod
    def get_all_topics(cls) -> List[str]:
        """Get list of all topic IDs"""
        return list(cls.SEEDS.keys())

    @classmethod
    def get_seeds_for_topic(cls, topic_id: str) -> List[str]:
        """Get seed pages for a specific topic"""
        if topic_id not in cls.SEEDS:
            raise ValueError(f"Unknown topic_id: {topic_id}")
        return cls.SEEDS[topic_id]

    @classmethod
    def validate(cls):
        """Validate seed configuration"""
        if not cls.SEEDS:
            raise RuntimeError("SEEDS dictionary is empty")

        for topic_id, seeds in cls.SEEDS.items():
            if not seeds:
                raise RuntimeError(f"Topic {topic_id} has no seed pages")

            if not all(isinstance(s, str) for s in seeds):
                raise RuntimeError(f"Topic {topic_id} contains non-string seeds")
