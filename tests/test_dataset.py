from dataset import SAMPLE_POSTS, TRUE_LABELS

ALLOWED_LABELS = {"positive", "negative", "neutral", "mixed"}


def test_posts_and_labels_are_aligned():
    assert len(SAMPLE_POSTS) == len(TRUE_LABELS)


def test_dataset_is_not_empty():
    assert len(SAMPLE_POSTS) > 0


def test_labels_are_from_the_allowed_set():
    for label in TRUE_LABELS:
        assert label in ALLOWED_LABELS
