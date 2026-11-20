"""Country and currency: guessed from the phone, spoken in words."""

from __future__ import annotations

import re

CURRENCY_WORDS = {
    "INR": "rupees",
    "USD": "dollars",
    "GBP": "pounds",
    "EUR": "euros",
    "CAD": "Canadian dollars",
    "AUD": "Australian dollars",
}
COUNTRY_NAMES = {
    "IN": "India",
    "US": "the United States",
    "GB": "the United Kingdom",
    "DE": "Germany",
    "FR": "France",
    "CA": "Canada",
    "AU": "Australia",
}
TZ_COUNTRY = {
    "Asia/Kolkata": "IN",
    "Asia/Calcutta": "IN",
    "Europe/London": "GB",
    "Europe/Berlin": "DE",
    "Europe/Paris": "FR",
    "Australia/Sydney": "AU",
}


def guess_country(locale: str, tz: str) -> str:
    """ISO country from the time zone of the phone, else from its locale; empty when neither says."""
    if tz in TZ_COUNTRY:
        return TZ_COUNTRY[tz]
    m = re.search(r"[-_]([A-Z]{2})$", locale)
    region = m.group(1) if m else ""
    # Many countries share America/* zones: the phone's language region decides; plain "en" counts as the US.
    if tz.startswith("America/") and region in ("", "US"):
        return "US"
    return region


_YES = re.compile(
    r"\b(yes|yeah|yep|yup|ya|correct|right|sure|okay|ok|absolutely|exactly|fine|perfect|haan|confirmed?)\b"
)
_NO = re.compile(r"\b(no|not|nope|wrong|actually|instead)\b")


def confirms(text: str) -> bool:
    """A short yes to "is that right?" ("Yes", "I'm in India, rupees is right"). A correction is not one."""
    t = text.lower()
    return len(t.split()) <= 10 and bool(_YES.search(t)) and not _NO.search(t)


_NAMED = {
    **{name.lower().removeprefix("the "): code for code, name in COUNTRY_NAMES.items()},
    "usa": "US",
    "america": "US",
    "uk": "GB",
    "england": "GB",
    "britain": "GB",
}


def country_named(text: str) -> str:
    """The country a sentence names ("I'm in India", "we're in the UK"), or empty."""
    t = text.lower()
    for name, code in sorted(_NAMED.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name)}\b", t):
            return code
    return ""
