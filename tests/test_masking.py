"""Tests for the PII masking middleware (src/masking.py)."""

import logging

import pytest

from masking import PIIMasker, PIIRedactingFilter, redact


@pytest.mark.parametrize(
    "phone",
    [
        "0712345678",          # 07xx, no spaces
        "0712 345 678",        # 07xx, spaced
        "0712-345-678",        # 07xx, hyphenated
        "0110345678",          # 01xx new range
        "0110 345 678",        # 01xx spaced
        "+254712345678",       # +254, no spaces
        "+254 712 345 678",    # +254 spaced
        "254712345678",        # 254 without plus
        "254 712 345 678",     # 254 spaced
        "+254110345678",       # +254 with 1xx range
        "+254 110 345 678",
    ],
)
def test_masks_every_kenyan_phone_format(phone):
    masker = PIIMasker()
    masked = masker.mask(f"Call me on {phone} today.")
    assert phone not in masked
    assert masked == "Call me on [MASKED_PHONE_1] today."
    assert masker.vault["[MASKED_PHONE_1]"] == phone


@pytest.mark.parametrize(
    "email",
    ["jane.doe@example.com", "w.kamau+claims@afyaplus.co.ke", "PATIENT_01@Mail.Example.org"],
)
def test_masks_emails(email):
    masker = PIIMasker()
    masked = masker.mask(f"Email {email} please")
    assert masked == "Email [MASKED_EMAIL_1] please"
    assert masker.vault["[MASKED_EMAIL_1]"] == email


def test_full_stop_after_email_is_not_captured():
    masker = PIIMasker()
    masked = masker.mask("Reach me at jane@afyaplus.co.ke.")
    assert masked == "Reach me at [MASKED_EMAIL_1]."
    assert masker.vault["[MASKED_EMAIL_1]"] == "jane@afyaplus.co.ke"


def test_phone_followed_by_punctuation():
    masker = PIIMasker()
    assert masker.mask("My line is 0722 000 111.") == "My line is [MASKED_PHONE_1]."


def test_mixed_pii_gets_separate_numbered_tokens():
    masker = PIIMasker()
    text = "Phones 0712 345 678 and +254 733 111 222, email a@b.co.ke."
    masked = masker.mask(text)
    assert masked == "Phones [MASKED_PHONE_1] and [MASKED_PHONE_2], email [MASKED_EMAIL_1]."


def test_same_number_in_different_formats_reuses_token():
    masker = PIIMasker()
    first = masker.mask("0712 345 678")
    second = masker.mask("+254712345678")
    assert first == second == "[MASKED_PHONE_1]"


def test_digits_inside_email_are_treated_as_email_not_phone():
    masker = PIIMasker()
    assert masker.mask("0712345678@gmail.com") == "[MASKED_EMAIL_1]"


@pytest.mark.parametrize(
    "safe_text",
    [
        "Policy number 1234567890",      # 10 digits but not starting 07/01
        "Dose is 500 mg every 8 hours",
        "Invoice 07123456789012",        # too long to be a phone
        "Weight 18 kg, height 110 cm",
    ],
)
def test_does_not_mask_non_pii_numbers(safe_text):
    assert PIIMasker().mask(safe_text) == safe_text


def test_unmask_restores_original_text():
    masker = PIIMasker()
    original = "I'm on 0712 345 678 or wanjiru@example.co.ke."
    masked = masker.mask(original)
    assert "0712" not in masked and "@" not in masked
    assert masker.unmask(masked) == original


def test_unmask_leaves_unknown_tokens_untouched():
    assert PIIMasker().unmask("See [MASKED_PHONE_9]") == "See [MASKED_PHONE_9]"


def test_redact_and_logging_filter_strip_pii(caplog):
    assert redact("x 0712345678 y a@b.com") == "x [REDACTED_PHONE] y [REDACTED_EMAIL]"

    logger = logging.getLogger("afyaplus.test")
    logger.addFilter(PIIRedactingFilter())
    with caplog.at_level(logging.INFO, logger="afyaplus.test"):
        logger.info("Patient %s called from %s", "a@b.com", "+254 712 345 678")
    assert "a@b.com" not in caplog.text
    assert "712" not in caplog.text
    assert "[REDACTED_EMAIL]" in caplog.text
