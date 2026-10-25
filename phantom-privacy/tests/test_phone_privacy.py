"""Tests for the phone-privacy guidance and VoIP comparison table."""

from phantom_privacy import phone_privacy


def test_render_includes_all_topic_titles():
    text = phone_privacy.render()
    for topic in phone_privacy.ALL_TOPICS:
        assert topic.title in text


def test_voip_table_lists_every_service():
    table = phone_privacy.render_voip_table()
    for service in phone_privacy.VOIP_SERVICES:
        assert service.name in table


def test_voip_table_has_headers_and_rows_aligned():
    table = phone_privacy.render_voip_table()
    lines = table.splitlines()
    assert "Service" in lines[0]
    assert "Privacy" in lines[0]
    # Header, separator and one line per service, at minimum.
    assert len(lines) >= 2 + len(phone_privacy.VOIP_SERVICES)


def test_voip_table_custom_subset():
    one = phone_privacy.VOIP_SERVICES[:1]
    table = phone_privacy.render_voip_table(one)
    assert one[0].name in table
    # A service not in the subset should not appear.
    others = [s for s in phone_privacy.VOIP_SERVICES if s.name != one[0].name]
    for s in others:
        assert s.name not in table
