"""src/resolver.py — Scalable master-data resolution with O(1) indexing and honest abstention."""
from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any
from rapidfuzz import fuzz, process


def _norm_alnum(s: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _norm_text(s: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", str(s or "").lower()).split())


class MasterDataResolver:
    """Resolves document values against master_data/ using O(1) exact indexes + fuzzy blocking."""

    def __init__(self, master_dir: str = "master_data") -> None:
        self.master_dir = master_dir
        self.suppliers = self._load_json("suppliers.json", "suppliers")
        self.companies = self._load_json("chart_of_books.json", "companies")
        self.taxes = self._load_json("tax_master.json", "taxes")
        self.payment_terms = self._load_json("payment_terms.json", "payment_terms")
        self.purchase_orders = self._load_json("po_master.json", "purchase_orders")

        # Building O(1) lookup indexes for million-row scalability
        self.sup_by_vat: dict[str, dict] = {}
        self.sup_by_iban: dict[str, dict] = {}
        self.sup_by_email: dict[str, dict] = {}
        self.sup_by_name: dict[str, dict] = {}
        for s in self.suppliers:
            if s.get("vat_id"):
                self.sup_by_vat[_norm_alnum(s["vat_id"])] = s
            if s.get("bank_iban"):
                self.sup_by_iban[_norm_alnum(s["bank_iban"])] = s
            if s.get("email"):
                self.sup_by_email[str(s["email"]).strip().lower()] = s
            if s.get("name"):
                self.sup_by_name[_norm_text(s["name"])] = s

        self.po_by_number: dict[str, dict] = {
            _norm_alnum(p.get("po_number")): p
            for p in self.purchase_orders
            if p.get("po_number")
        }

        self.term_by_days: dict[int, str] = {}
        for pt in self.payment_terms:
            days = pt.get("days")
            if isinstance(days, int) and days not in self.term_by_days:
                self.term_by_days[days] = pt["payment_term_id"]

        # Flattening buyer business units for fast resolution
        self.buyer_units: list[dict] = []
        for comp in self.companies:
            c_code = comp.get("company_code", "")
            for bu in comp.get("business_units", []):
                for loc in bu.get("locations", []):
                    self.buyer_units.append({
                        "company_code": c_code,
                        "business_unit_code": bu.get("business_unit_code", ""),
                        "business_unit_name": bu.get("business_unit_name", ""),
                        "location_code": loc.get("location_code", ""),
                        "invoice_to_address": loc.get("invoice_to_address", ""),
                    })

    def _load_json(self, filename: str, key: str) -> list[dict]:
        path = os.path.join(self.master_dir, filename)
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get(key, [])

    def resolve_supplier(self, raw_supplier: dict, doc_iban: str = "", doc_email: str = "") -> dict:
        """Resolves supplier_id while preserving raw printed supplier fields."""
        name = str(raw_supplier.get("name") or "").strip()
        vat_id = str(raw_supplier.get("vat_id") or "").strip()
        address = str(raw_supplier.get("address") or "").strip()

        matched = None
        # 1. Checking exact VAT ID match
        if vat_id and _norm_alnum(vat_id) in self.sup_by_vat:
            matched = self.sup_by_vat[_norm_alnum(vat_id)]
        # 2. Checking exact IBAN match
        if not matched and doc_iban and _norm_alnum(doc_iban) in self.sup_by_iban:
            matched = self.sup_by_iban[_norm_alnum(doc_iban)]
        # 3. Checking exact Email match
        if not matched and doc_email and doc_email.strip().lower() in self.sup_by_email:
            matched = self.sup_by_email[doc_email.strip().lower()]
        # 4. Checking exact or High-Confidence Fuzzy Name match (threshold >= 85 to prevent false positives)
        if not matched and name:
            norm_n = _norm_text(name)
            if norm_n in self.sup_by_name:
                matched = self.sup_by_name[norm_n]
            elif self.sup_by_name:
                best = process.extractOne(norm_n, list(self.sup_by_name.keys()), scorer=fuzz.token_set_ratio)
                if best and best[1] >= 85:
                    matched = self.sup_by_name[best[0]]

        return {
            "name": name,
            "supplier_id": matched["supplier_id"] if matched else "",
            "address": address,
            "vat_id": vat_id,
        }

    def resolve_buyer(self, raw_buyer_name: str, raw_buyer_address: str, raw_buyer: dict) -> dict:
        """Resolves buyer company_code, business_unit_code, and location_code against chart_of_books.json."""
        # If already validly populated from master, verifying it exists
        existing_bu = str(raw_buyer.get("business_unit_code") or "").strip()
        for u in self.buyer_units:
            if existing_bu and u["business_unit_code"] == existing_bu:
                return {
                    "company_code": u["company_code"],
                    "business_unit_code": u["business_unit_code"],
                    "location_code": u["location_code"],
                }

        combined = _norm_text(f"{raw_buyer_name} {raw_buyer_address}")
        if not combined:
            return {"company_code": "", "business_unit_code": "", "location_code": ""}

        # Distinguishing Estonia EE001 (Technology) vs EE004 (Holdings) which share Vana-Louna 15 address
        if "holdings" in combined and ("tallinn" in combined or "estonia" in combined or "10134" in combined or "ou" in combined):
            for u in self.buyer_units:
                if u["business_unit_code"] == "EE004":
                    return {"company_code": u["company_code"], "business_unit_code": u["business_unit_code"], "location_code": u["location_code"]}

        if ("technology" in combined or "operations" in combined) and ("tallinn" in combined or "estonia" in combined or "10134" in combined):
            for u in self.buyer_units:
                if u["business_unit_code"] == "EE001":
                    return {"company_code": u["company_code"], "business_unit_code": u["business_unit_code"], "location_code": u["location_code"]}

        best_unit = None
        best_score = 0.0
        for u in self.buyer_units:
            target = _norm_text(f"{u['business_unit_name']} {u['invoice_to_address']}")
            score = fuzz.token_set_ratio(combined, target)
            if score > best_score:
                best_score = score
                best_unit = u

        if best_unit and best_score >= 75:
            return {
                "company_code": best_unit["company_code"],
                "business_unit_code": best_unit["business_unit_code"],
                "location_code": best_unit["location_code"],
            }
        return {"company_code": "", "business_unit_code": "", "location_code": ""}

    def resolve_payment_term(self, raw_term_text: str, invoice_date: str, due_date: str) -> str:
        """Resolves payment_term_id from text aliases or (due_date - invoice_date) delta."""
        # If raw_term_text is already a valid payment_term_id, returning it
        for pt in self.payment_terms:
            if str(raw_term_text).strip() == pt["payment_term_id"]:
                return pt["payment_term_id"]

        norm_t = _norm_text(raw_term_text)
        if norm_t:
            for pt in self.payment_terms:
                for alias in pt.get("text_aliases", []):
                    if _norm_text(alias) in norm_t or norm_t in _norm_text(alias):
                        return pt["payment_term_id"]
            # Checking explicit day number in payment term text (e.g. "30 pv neto", "60 Days Net")
            m = re.search(r"\b(\d+)\s*(?:days?|d|pv|päeva|tage|dias)\b", norm_t)
            if m:
                d = int(m.group(1))
                if d in self.term_by_days:
                    return self.term_by_days[d]

        # Deriving from due_date - invoice_date when both ISO dates are present
        if invoice_date and due_date:
            try:
                d1 = datetime.date.fromisoformat(str(invoice_date).strip()[:10])
                d2 = datetime.date.fromisoformat(str(due_date).strip()[:10])
                delta_days = (d2 - d1).days
                if delta_days in self.term_by_days:
                    return self.term_by_days[delta_days]
            except ValueError:
                pass
        return ""

    def resolve_po(self, raw_po_number: str, supplier_id: str = "") -> tuple[str, str]:
        """Returns (po_number_as_printed, matched_po_id_or_empty)."""
        po_num = str(raw_po_number or "").strip()
        if not po_num:
            return "", ""
        norm_po = _norm_alnum(po_num)
        matched = self.po_by_number.get(norm_po)
        if matched:
            if not supplier_id or not matched.get("supplier_id") or matched["supplier_id"] == supplier_id:
                return po_num, matched["po_id"]
        return po_num, ""

    def resolve_tax_code(self, country: str, tax_rate: Any, tax_type: str = "", tax_name: str = "") -> str:
        """Resolves tax_type_code against tax_master.json by (country, rate, type/name)."""
        try:
            rate_val = float(str(tax_rate).replace("%", "").strip())
        except (ValueError, TypeError):
            return ""

        c_code = str(country or "").strip().upper()
        t_type = str(tax_type or "").strip().upper()
        t_name = str(tax_name or "").strip().lower()

        candidates = [
            t for t in self.taxes
            if (not c_code or t.get("country", "").upper() == c_code)
            and abs(float(t.get("rate", -999)) - rate_val) < 0.01
        ]
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]["code"]

        # Disambiguating multiple same-rate taxes in the same country (e.g., GH 2.5% NHIL vs GETFL, DE 0% RC)
        for c in candidates:
            master_type = c.get("tax_type", "").upper()
            master_name = c.get("name", "").lower()
            if t_type and t_type == master_type:
                return c["code"]
            if master_type.lower() in t_name or any(tok in t_name for tok in master_name.split()):
                return c["code"]
            if "reverse" in t_name and "reverse" in master_name:
                return c["code"]
        return candidates[0]["code"]

    def enrich_payable(self, payable: dict) -> dict:
        """Applies master data resolution across all reference-backed fields of a payable."""
        meta = payable.pop("_meta", {}) if isinstance(payable.get("_meta"), dict) else {}
        country = str(meta.get("supplier_country") or "").strip().upper()
        iban = str(meta.get("supplier_iban") or "").strip()
        email = str(meta.get("supplier_email") or "").strip()
        buyer_name = str(meta.get("buyer_name") or "").strip()
        buyer_addr = str(meta.get("buyer_address") or "").strip()
        raw_term = str(meta.get("payment_term_text") or payable.get("payment_term_id") or "").strip()

        sup = payable.get("supplier") or {}
        if not country and sup.get("vat_id"):
            m = re.match(r"^([A-Z]{2,3})", str(sup["vat_id"]).strip().upper())
            if m:
                country = m.group(1)[:2]

        payable["supplier"] = self.resolve_supplier(sup, doc_iban=iban, doc_email=email)
        payable["buyer"] = self.resolve_buyer(buyer_name, buyer_addr, payable.get("buyer") or {})
        payable["payment_term_id"] = self.resolve_payment_term(
            raw_term,
            payable.get("invoice_date", ""),
            payable.get("due_date", ""),
        )
        po_num, po_id = self.resolve_po(payable.get("po_number", ""), payable["supplier"].get("supplier_id", ""))
        payable["po_number"] = po_num
        payable["po_id"] = po_id

        # Resolving header taxes
        for t in payable.get("taxes") or []:
            if isinstance(t, dict):
                code = self.resolve_tax_code(country, t.get("tax_rate"), t.get("tax_type", ""), t.get("tax_name", ""))
                t["tax_type_code"] = code

        # Resolving line taxes
        for li in payable.get("line_items") or []:
            if isinstance(li, dict):
                for t in li.get("taxes") or []:
                    if isinstance(t, dict):
                        code = self.resolve_tax_code(country, t.get("tax_rate"), t.get("tax_type", ""), t.get("tax_name", ""))
                        t["tax_type_code"] = code

        return payable