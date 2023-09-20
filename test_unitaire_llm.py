import pytest
from llm import callLlm

@pytest.mark.parametrize("q1, url, expected", [
    ("Quelle est la capitale de la France ?", "https://fr.wikipedia.org/wiki/Paris", "Paris"),
    ("Quel est le jour de la semaine aujourd'hui ?", "https://fr.wikipedia.org/wiki/Jour_de_la_semaine", "Mercredi"),
    ("Quelle est la population de la France ?", "https://fr.wikipedia.org/wiki/France", "67,34 millions"),
])
def test_callLlm(q1, url, expected):
    # Arrange

    # Act
    result = callLlm(q1, url)

    print(result)

    # Assert
    assert result == expected
