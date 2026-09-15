import pytest

from forecaster.grounding import check_rationale, domain_of


def test_clean_rationale_keeps_full_weight():
    report = check_rationale(
        "Base rates suggest No. Nothing in the evidence shows a scheduled announcement.",
        evidence_urls=["https://example.com/a"],
    )
    assert report.flags == []
    assert report.weight == 1.0


def test_tool_claims_are_flagged():
    report = check_rationale("I searched the news and found nothing new.", evidence_urls=[])
    assert any(f.startswith("TOOL_CLAIM") for f in report.flags)
    assert report.weight < 1.0


def test_unseen_sources_are_flagged_but_evidence_subdomains_are_allowed():
    evidence = ["https://www.nhc.noaa.gov/gtwo.php"]
    ok = check_rationale("Per https://nhc.noaa.gov/text/MIATWOAT.shtml the outlook is quiet.", evidence)
    assert ok.flags == []
    bad = check_rationale("Bloomberg (https://www.bloomberg.com/news/x) reports a delay.", evidence)
    assert any("bloomberg.com" in f for f in bad.flags)


def test_price_mismatch_requires_every_price_claim_to_be_off():
    far = check_rationale(
        "ZEC is currently trading at $900, well below the threshold.", [], reference_price=1130.0
    )
    assert any(f.startswith("PRICE_MISMATCH") for f in far.flags)
    near = check_rationale("ZEC is currently trading at $1,125.", [], reference_price=1130.0)
    assert near.flags == []


def test_threshold_mentions_are_not_mistaken_for_price_claims():
    report = check_rationale(
        "The price of $1,100 is the threshold, and spot is close to it.",
        [],
        reference_price=1130.0,
        known_values=[1100.0],
    )
    assert report.flags == []


def test_penalties_compound_down_to_a_floor():
    text = "I searched https://unknown.example and it is currently trading at $1."
    report = check_rationale(text, [], reference_price=1000.0)
    assert len(report.flags) == 3
    assert report.weight == pytest.approx(0.7 * 0.7 * 0.6)
    floored = check_rationale(text, [], reference_price=1000.0, min_weight=0.5)
    assert floored.weight == 0.5


def test_domain_of_strips_www_and_ports():
    assert domain_of("https://www.example.com:8443/path") == "example.com"
