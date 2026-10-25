"""Tests for the footprint scoring logic."""

from phantom_privacy import footprint


def test_all_good_answers_give_full_score():
    answers = {q.key: q.good_answer for q in footprint.QUESTIONS}
    result = footprint.assess(answers)
    assert result.score == result.max_score
    assert result.percent == 100
    assert result.rating == "Strong"
    assert result.recommendations == []


def test_all_bad_answers_give_zero_and_all_advice():
    answers = {q.key: (not q.good_answer) for q in footprint.QUESTIONS}
    result = footprint.assess(answers)
    assert result.score == 0
    assert result.percent == 0
    assert result.rating == "Exposed"
    assert len(result.recommendations) == len(footprint.QUESTIONS)


def test_skipped_questions_do_not_count():
    # Answer only one question, well.
    first = footprint.QUESTIONS[0]
    result = footprint.assess({first.key: first.good_answer})
    assert result.max_score == first.weight
    assert result.score == first.weight
    assert result.percent == 100


def test_partial_mixed_answers():
    questions = [
        footprint.Question("a", "?", good_answer=True, weight=2, advice="fix a"),
        footprint.Question("b", "?", good_answer=False, weight=3, advice="fix b"),
    ]
    # a answered well (+2), b answered badly (advice)
    result = footprint.assess({"a": True, "b": True}, questions=questions)
    assert result.score == 2
    assert result.max_score == 5
    assert result.recommendations == ["fix b"]
    assert result.percent == 40
    assert result.rating == "Exposed"  # 40% is below the 50% "Fair" threshold


def test_rating_boundaries():
    q = [footprint.Question("x", "?", good_answer=True, weight=10, advice="a")]
    assert footprint.assess({"x": True}, q).rating == "Strong"   # 100%
    assert footprint.assess({"x": False}, q).rating == "Exposed"  # 0%
