from dataset import SAMPLE_POSTS
from mood_analyzer import MoodAnalyzer


def test_preprocess_lowercases_and_splits():
    analyzer = MoodAnalyzer()
    assert analyzer.preprocess("  Great Day  ") == ["great", "day"]


def test_custom_word_lists_override_the_defaults():
    analyzer = MoodAnalyzer(positive_words=["stoked"], negative_words=["meh"])
    assert "stoked" in analyzer.positive_words
    assert "happy" not in analyzer.positive_words


def test_explain_reports_the_words_it_matched():
    analyzer = MoodAnalyzer(positive_words=["great"], negative_words=["bad"])
    explanation = analyzer.explain("this is great and not bad")
    assert "great" in explanation
    assert "bad" in explanation


def test_predict_label_runs_without_crashing_on_every_sample_post():
    analyzer = MoodAnalyzer()
    for post in SAMPLE_POSTS:
        analyzer.predict_label(post)


def test_predict_label_matches_true_labels_for_easy_examples():
    analyzer = MoodAnalyzer()
    assert analyzer.predict_label("I love this class so much") == "positive"
    assert analyzer.predict_label("Today was a terrible day") == "negative"
    assert analyzer.predict_label("This is fine") == "neutral"


def test_predict_label_handles_simple_negation():
    analyzer = MoodAnalyzer()
    assert analyzer.predict_label("I am not happy about this") == "negative"
