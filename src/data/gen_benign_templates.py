# src/data/gen_benign_templates.py
from pathlib import Path
import pandas as pd

OUT_PATH = Path("data/manual_benign/templated_benign.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

topics = [
    "prompt injection attacks",
    "jailbreak prompts in large language models",
    "system prompts and hidden instructions",
    "safe AI guardrails for chatbots",
    "content filtering for LLM outputs",
    "data privacy in conversational AI",
    "access control in AI assistants",
    "role-based permissions in LLM applications",
    "logging and monitoring of AI systems",
    "input validation for user prompts",
    "output validation for AI responses",
    "rate limiting and abuse prevention in APIs",
    "secure prompt design for customer support bots",
    "secure handling of personally identifiable information",
    "red-teaming of language models",
    "security testing for AI-based systems",
    "SQL injection vs prompt injection",
    "cross-site scripting vs prompt injection",
    "phishing emails and social engineering attacks",
    "threat modeling for AI applications",
    "zero-trust principles in LLM deployments",
    "defense in depth for AI security",
    "secure coding practices for Python backends",
    "API key management and rotation",
    "secrets management in cloud environments",
    "least privilege principle for services",
    "incident response for AI security breaches",
    "monitoring anomalous user behavior in chatbots",
    "detecting unusual prompt patterns",
    "explainable AI in security workflows",
    "ethical use of LLMs in cybersecurity",
    "dataset curation for safe AI training",
    "labeling harmful vs harmless prompts",
    "benchmarking prompt injection defenses",
    "OWASP Top 10 for LLM applications",
    "supply chain risks in AI models",
    "model stolen prompt / prompt leakage",
    "red-team vs blue-team exercises with LLMs",
    "governance and policy for AI security",
    "user education about safe AI usage",
]

audiences = [
    "for a complete beginner",
    "for a non-technical manager",
    "for a security engineer",
    "for a student learning cybersecurity",
    "for a machine learning engineer",
]

styles = [
    "as a short paragraph",
    "as 3-4 bullet points",
]

question_patterns = [
    "Explain what {topic} is {audience}, {style}.",
    "Summarize the main ideas of {topic} {audience}, {style}.",
    "Give a simple real-world example of {topic} {audience}, {style}.",
    "List a few best practices related to {topic} {audience}, {style}.",
    "Describe why {topic} matters in real systems {audience}, {style}.",
    "Paraphrase a short explanation of {topic} {audience}, {style}.",
    "Compare {topic} with a more traditional security concept {audience}, {style}.",
    "Identify common mistakes people make when thinking about {topic} {audience}, {style}.",
    "Write a short FAQ about {topic} {audience}, {style}.",
    "Suggest safe questions a user could ask about {topic} {audience}, {style}.",
    "Explain how {topic} fits into the bigger picture of AI security {audience}, {style}.",
    "Summarize what a beginner should remember about {topic} {audience}, {style}.",
    "Explain {topic} without including any instructions that could be abused {audience}, {style}.",
    "Describe how to document {topic} clearly in an internal security guideline {audience}, {style}.",
    "Explain how you would teach {topic} in a short training session {audience}, {style}.",
]

rows = []
seen = set()

for topic in topics:
    for audience in audiences:
        for style in styles:
            for pattern in question_patterns:
                text = pattern.format(topic=topic, audience=audience, style=style)
                text = " ".join(text.split())
                if text not in seen:
                    seen.add(text)
                    rows.append({
                        "text": text,
                        "label": 0,  # benign
                        "source": "templated_benign_v2",
                    })

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False, encoding="utf-8")

print(f"✅ Saved benign template dataset: {OUT_PATH}")
print("✅ Rows:", len(df))
print(df.head())
