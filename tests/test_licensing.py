# SPDX-License-Identifier: Apache-2.0
import pytest

from lapsus.licensing import (LicenseError, combine_licenses, decide_mode, is_permissive, most_restrictive,
                              row_allowed)


def test_permissive_only_apache_cc0_pd():
    assert is_permissive("Apache-2.0") and is_permissive("CC0-1.0") and is_permissive("LicenseRef-public-domain")
    assert not is_permissive("CC-BY-2.0") and not is_permissive("CC-BY-SA-3.0") and not is_permissive("LicenseRef-unverified")


def test_combine_licenses_compatibility_table():
    assert combine_licenses({"CC0-1.0", "Apache-2.0"})["hf"] == "apache-2.0"
    assert combine_licenses({"CC0-1.0", "CC-BY-2.0", "Apache-2.0"})["hf"] == "cc-by-2.0"
    assert combine_licenses({"CC-BY-2.0", "CC-BY-SA-3.0"})["hf"] == "cc-by-sa-3.0"
    r = combine_licenses({"CC-BY-2.0", "CC-BY-NC-SA-3.0"})
    assert r["hf"] == "cc-by-nc-sa-3.0" and r["noncommercial"] and r["share_alike"]
    assert combine_licenses({"CC-BY-2.0", "CC-BY-NC-4.0"})["hf"] == "cc-by-nc-4.0"
    assert combine_licenses({"CC-BY-2.0", "LicenseRef-unverified"})["hf"] == "other"


def test_most_restrictive_across_configs():
    assert most_restrictive(["apache-2.0", "cc-by-2.0"]) == "cc-by-2.0"
    assert most_restrictive(["cc-by-2.0", "cc-by-sa-3.0", "apache-2.0"]) == "cc-by-sa-3.0"
    assert most_restrictive(["apache-2.0"]) == "apache-2.0"


def test_apache_strict_fails_when_pool_too_small():
    with pytest.raises(LicenseError):
        decide_mode("it", "apache-strict", 19, 30000, 15000)
    d = decide_mode("en", "apache-strict", 20000, 40000, 15000)
    assert d.mode == "apache-strict"


def test_source_licenses_mode():
    d = decide_mode("it", "source-licenses", 19, 30000, 15000)
    assert d.mode == "source-licenses"
    assert row_allowed("CC-BY-2.0", d.mode) and row_allowed("CC0-1.0", d.mode)


def test_auto_decides_per_language():
    en = decide_mode("en", "auto", 16000, 40000, 15000)
    it = decide_mode("it", "auto", 19, 30000, 15000)
    assert en.mode == "apache-strict" and "permissive pool" in en.reason
    assert it.mode == "source-licenses" and "falling back" in it.reason
    assert not row_allowed("CC-BY-2.0", en.mode) and row_allowed("CC0-1.0", en.mode)
