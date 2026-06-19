"""Phase 5 — build the golden dataset for the smolagents dogfood run.

A ~24-case multi-hop factual QA suite in the spirit of HotpotQA: each question
requires *bridging two facts*, so a good answer needs at least one web search and
often a page visit. Multi-hop is deliberate — single-hop lookups rarely expose
trajectory failures (wrong tool, hallucinated args, retrieval miss,
correct-retrieval-wrong-answer), which are exactly what Autopsy is built to catch.

Tool names match smolagents' defaults: `web_search`, `visit_webpage`
(plus the implicit `final_answer`).

Run:
    python examples/build_dogfood_dataset.py
    # -> writes examples/datasets/multihop_qa.json
"""

from __future__ import annotations

from pathlib import Path

from autopsy import Dataset, TestCase

# Tools the smolagents agent is allowed to use (default toolbox).
ALLOWED_TOOLS = ["web_search", "visit_webpage", "final_answer"]
# The minimal tool a correct trajectory must use.
EXPECTED_TOOLS = ["web_search"]

# (question, gold answer, number of reasoning hops)
QA: list[tuple[str, str, int]] = [
    ("The actor who played Iron Man in the Marvel Cinematic Universe was born in what city?",
     "Manhattan, New York City", 2),
    ("What is the nationality of the director of the film that won the Academy Award for Best Picture in 2020?",
     "South Korean", 2),
    ("The author of 'A Brief History of Time' held a university professorship previously held by which 17th-century scientist?",
     "Isaac Newton", 2),
    ("What is the capital of the country where the 2016 Summer Olympics were held?",
     "Brasilia", 2),
    ("The company that created the iPhone has its headquarters in which city?",
     "Cupertino", 2),
    ("Who was the U.S. president when the first person walked on the Moon?",
     "Richard Nixon", 2),
    ("The river that flows through Paris empties into which body of water?",
     "English Channel", 2),
    ("What language is primarily spoken in the country that borders both France and Slovenia?",
     "Italian", 2),
    ("The scientist who developed the theory of general relativity was awarded the Nobel Prize in Physics for which phenomenon?",
     "the photoelectric effect", 2),
    ("What currency is used in the country whose capital hosted the 1900 Summer Olympics?",
     "Euro", 2),
    ("The 'Mona Lisa' is displayed in a museum located in which country?",
     "France", 2),
    ("Who wrote the novel that inspired the film 'Blade Runner'?",
     "Philip K. Dick", 2),
    ("What is the capital of the country off whose coast the Great Barrier Reef lies?",
     "Canberra", 2),
    ("The physicist who formulated the three laws of motion was born in which century?",
     "17th century", 2),
    ("Curium is named after a scientist who was born in which city?",
     "Warsaw", 2),
    ("The director of 'Inception' also directed a Batman trilogy starring which actor as Batman?",
     "Christian Bale", 2),
    ("What is the official language of the country where the headquarters of the United Nations is located?",
     "English", 2),
    ("The composer of the symphony that includes 'Ode to Joy' was from which country?",
     "Germany", 2),
    ("What is the currency of the country where the world's tallest building is located?",
     "United Arab Emirates dirham", 2),
    ("The co-founder of Microsoft who served as its first CEO attended which university before dropping out?",
     "Harvard University", 2),
    ("The country that won the 2014 FIFA World Cup is located on which continent?",
     "Europe", 2),
    ("What is the largest moon of the largest planet in our solar system?",
     "Ganymede", 2),
    ("The painter of 'The Starry Night' was born in which country?",
     "Netherlands", 2),
    ("The actor who voiced Woody in 'Toy Story' also starred in a film about a man stranded on a desert island; what is that film's name?",
     "Cast Away", 2),
]


def build() -> Dataset:
    cases = [
        TestCase(
            input=question,
            expected_output=answer,
            expected_tools=EXPECTED_TOOLS,
            allowed_tools=ALLOWED_TOOLS,
            # Budget counts every span (LLM calls + tool calls). A clean 2-hop
            # run is ~5-6 spans; 8 leaves headroom so only wasteful runs trip it.
            max_steps=8,
            metadata={"hops": hops, "source": "HotpotQA-style multi-hop (hand-authored, stable facts)"},
        )
        for question, answer, hops in QA
    ]
    return Dataset(
        name="multihop_qa",
        version="1.0.0",
        description=(
            "Multi-hop factual QA for dogfooding a tool-using agent. Each question "
            "bridges two facts, requiring search (and often a page visit) to answer."
        ),
        source="hand-authored, HotpotQA-style; answers chosen to be stable over time",
        cases=cases,
    )


def main() -> None:
    dataset = build()
    out = Path(__file__).parent / "datasets" / "multihop_qa.json"
    dataset.save(out)
    print(f"Wrote {len(dataset)} cases -> {out}")


if __name__ == "__main__":
    main()
