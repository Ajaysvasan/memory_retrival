from typing import Any, Dict


class PromptFormatter:
    """Format structured prompts."""

    @staticmethod
    def format(query: str, context: Dict[str, Any]) -> str:
        """Create structured prompt."""
        sections = []

        if context.get("constraints"):
            sections.append("CONSTRAINTS:\n" + "\n".join(context["constraints"]))

        if context.get("recent_turns"):
            sections.append(
                "RECENT CONVERSATION:\n"
                + "\n".join(
                    [
                        f"Q: {t['query']}\nA: {t['response']}"
                        for t in context["recent_turns"]
                    ]
                )
            )

        if context.get("retrieved"):
            sections.append("EVIDENCE:\n" + "\n\n".join(context["retrieved"]))

        sections.append(f"QUERY: {query}")
        sections.append("RESPONSE:")

        return "\n\n".join(sections)
