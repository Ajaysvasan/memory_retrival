from typing import List, Set
import logging


class TopicAssigner:
    """Assign primary and secondary topics to pages"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TopicAssigner")

        self.topic_keywords = {
            "cardiovascular": {
                "heart",
                "cardiac",
                "vascular",
                "arterial",
                "coronary",
                "myocardial",
            },
            "cancer": {
                "cancer",
                "carcinoma",
                "tumor",
                "malignant",
                "oncology",
                "metastasis",
            },
            "respiratory": {"lung", "pulmonary", "respiratory", "breathing", "asthma"},
            "diabetes": {"diabetes", "diabetic", "insulin", "glucose", "glycemic"},
            "infectious_disease": {
                "infection",
                "infectious",
                "pathogen",
                "bacteria",
                "virus",
                "viral",
            },
            "mental_health": {
                "mental",
                "psychiatric",
                "psychological",
                "depression",
                "anxiety",
            },
            "public_health": {
                "public health",
                "prevention",
                "health policy",
                "epidemiology",
            },
        }

    def assign(self, extracted_data: dict) -> dict:
        """Assign topics to extracted page data"""

        primary_topic = extracted_data["topic_id"]

        secondary_topics = self._find_secondary_topics(
            extracted_data, exclude=primary_topic
        )

        assigned = extracted_data.copy()
        assigned["primary_topic_id"] = primary_topic
        assigned["secondary_topics"] = list(secondary_topics)

        self.logger.debug(
            f"{extracted_data['title']}: primary={primary_topic}, "
            f"secondary={secondary_topics}"
        )

        return assigned

    def _find_secondary_topics(self, data: dict, exclude: str) -> Set[str]:
        """Find secondary topics based on content"""

        text_to_analyze = data["title"].lower()

        for section in data.get("sections", []):
            text_to_analyze += " " + section.get("heading", "").lower()
            text_to_analyze += " " + section.get("text", "")[:500].lower()

        secondary = set()

        for topic_id, keywords in self.topic_keywords.items():
            if topic_id == exclude:
                continue

            if any(keyword in text_to_analyze for keyword in keywords):
                secondary.add(topic_id)

        return secondary
