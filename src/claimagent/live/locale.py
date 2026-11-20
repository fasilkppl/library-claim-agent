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
