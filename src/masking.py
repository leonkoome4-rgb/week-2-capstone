"""Privacy-masking middleware for Kenyan PII (phone numbers and emails).

Raw user text is passed through `PIIMasker.mask()` BEFORE it reaches any AI model.
Real values are swapped for placeholder tokens such as [MASKED_PHONE_1] and kept in a
local, in-memory vault. `PIIMasker.unmask()` re-injects them into the approved answer
right before it is shown to the user (compliance de-masking output layer).

Supports the data-minimisation principle of the Kenya Data Protection Act, 2019 (s.25):
the model only ever sees the minimum data needed to do its job.
"""

from __future__ import annotations

import logging
import re

# --- Patterns -----------------------------------------------------------------

# Email: the domain must END with letters (e.g. ".co.ke"), so a full stop that
# closes a sentence ("...me at jane@afya.co.ke.") is never captured.
EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])"                     # not glued to another email-ish character
    r"[A-Za-z0-9._%+-]+"                # local part
    r"@"
    r"(?:[A-Za-z0-9-]+\.)+"             # one or more domain labels ending in a dot
    r"[A-Za-z]{2,}"                     # top-level domain: letters only
    r"(?![A-Za-z0-9-])"
)

# Kenyan mobile / new-range numbers:
#   07xx xxx xxx | 01xx xxx xxx | +254 7xx xxx xxx | 254 7xx xxx xxx | +254 1xx ...
# Groups may be separated by a single space or hyphen, or written together.
PHONE_PATTERN = re.compile(
    r"(?<![\w+])"                       # not in the middle of a longer number/word
    r"(?:\+?254[\s-]?|0)"               # country code (+254 / 254) or trunk prefix 0
    r"[17]\d{2}"                        # network prefix: 7xx (mobile) or 1xx (new range)
    r"[\s-]?\d{3}"
    r"[\s-]?\d{3}"
    r"(?!\d)"                           # no extra trailing digits
)

TOKEN_PATTERN = re.compile(r"\[MASKED_(?:PHONE|EMAIL)_\d+\]")


class PIIMasker:
    """Masks and unmasks PII for ONE conversation (one patient / thread).

    The same real value always maps to the same token, so multi-turn memory stays
    consistent: if a patient repeats their number in turn 3, the model still sees
    [MASKED_PHONE_1].
    """

    def __init__(self) -> None:
        self.vault: dict[str, str] = {}          # token -> real value (stays local)
        self._reverse: dict[str, str] = {}       # real value -> token
        self._counters = {"PHONE": 0, "EMAIL": 0}

    def _token_for(self, kind: str, value: str) -> str:
        key = _normalise(kind, value)
        if key in self._reverse:
            return self._reverse[key]
        self._counters[kind] += 1
        token = f"[MASKED_{kind}_{self._counters[kind]}]"
        self.vault[token] = value
        self._reverse[key] = token
        return token

    def mask(self, text: str) -> str:
        """Replace every email and Kenyan phone number in `text` with a token.

        Emails are masked first so digits inside an address
        (e.g. 0712345678@mail.com) are not mistaken for a phone number.
        """
        if not text:
            return text
        text = EMAIL_PATTERN.sub(lambda m: self._token_for("EMAIL", m.group()), text)
        text = PHONE_PATTERN.sub(lambda m: self._token_for("PHONE", m.group()), text)
        return text

    def unmask(self, text: str) -> str:
        """Put the original values back in place of their tokens."""
        if not text:
            return text
        return TOKEN_PATTERN.sub(lambda m: self.vault.get(m.group(), m.group()), text)


def _normalise(kind: str, value: str) -> str:
    """Canonical form so '0712 345 678' and '+254712345678' share one token."""
    if kind == "EMAIL":
        return value.lower()
    digits = re.sub(r"\D", "", value)
    if digits.startswith("254"):
        digits = "0" + digits[3:]
    return digits


def redact(text: str) -> str:
    """One-way redaction (no vault) for logs and error messages."""
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    return PHONE_PATTERN.sub("[REDACTED_PHONE]", text)


class PIIRedactingFilter(logging.Filter):
    """Logging filter: guarantees PII never lands in log files ("mask at every exit")."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True
