"""src/reconciler.py — Algebraic verification against erp.py and target-less diagnostic signals."""
from __future__ import annotations

import copy
import re
from typing import Any

from erp import erp_book, num, round2, _line_base, _line_taxes, _header_taxes


SCHEMA_TOP_KEYS = [
    "invoice_number", "invoice_date", "due_date", "invoice_type", "currency",
    "supplier", "buyer", "payment_term_id", "po_number", "po_id",
    "gross_total", "subtotal", "total_tax_amount",
    "discount_amount", "freight_charges", "insurance_charges", "extra_charges", "excise_duties",
    "taxes", "line_items",
]


def clean_num_str(val: Any) -> str:
    """Normalizes numeric strings to plain dot-decimal so erp.py num() never silently returns 0.0."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    s = re.sub(r"[€$£¥₹%\s\u00a0]", "", s)
    if not s:
        return ""
    if s.endswith("-") and not s.startswith("-"):
        s = "-" + s[:-1]
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and (len(parts[1]) in (1, 2) or parts[0].lstrip("-") == "0"):
            s = parts[0] + "." + parts[1]
        else:
            s = s.replace(",", "")
    try:
        float(s)
        return s
    except ValueError:
        return ""


def sanitize_payable_schema(p: dict) -> dict:
    """Ensures exact conformance to AUTODRAFT_SCHEMA.md types and dot-decimal numbers."""
    out = copy.deepcopy(p)
    out["invoice_type"] = "CREDIT_MEMO" if "CREDIT" in str(out.get("invoice_type", "")).upper() else "INVOICE"
    is_credit = out["invoice_type"] == "CREDIT_MEMO"

    num_fields = [
        "gross_total", "subtotal", "total_tax_amount",
        "discount_amount", "freight_charges", "insurance_charges", "extra_charges", "excise_duties",
    ]
    for nf in num_fields:
        c = clean_num_str(out.get(nf, ""))
        if is_credit and c.startswith("-"):
            c = c[1:]
        out[nf] = c

    if out.get("discount_amount", "").startswith("-"):
        out["discount_amount"] = out["discount_amount"][1:]

    clean_taxes = []
    for t in out.get("taxes") or []:
        if not isinstance(t, dict):
            continue
        rate = clean_num_str(t.get("tax_rate", ""))
        amt = clean_num_str(t.get("tax_amount", ""))
        if is_credit and amt.startswith("-") and "wht" not in str(t.get("tax_type", "")).lower():
            amt = amt[1:]
        clean_taxes.append({
            "tax_type": str(t.get("tax_type") or "VAT").strip(),
            "tax_name": str(t.get("tax_name") or "").strip(),
            "tax_rate": rate.lstrip("-"),
            "tax_amount": amt,
            "tax_type_code": str(t.get("tax_type_code") or "").strip(),
        })
    out["taxes"] = clean_taxes

    clean_lines = []
    for li in out.get("line_items") or []:
        if not isinstance(li, dict):
            continue
        qty = clean_num_str(li.get("quantity", ""))
        uprice = clean_num_str(li.get("unit_price", ""))
        ltot = clean_num_str(li.get("total", ""))
        disc = clean_num_str(li.get("discount", "")).lstrip("-")
        disc_pct = clean_num_str(li.get("discount_percentage", "")).lstrip("-")
        trate = clean_num_str(li.get("tax_rate", "")).lstrip("-")
        tamt = clean_num_str(li.get("tax_amount", ""))

        if is_credit:
            qty = qty.lstrip("-")
            uprice = uprice.lstrip("-")
            ltot = ltot.lstrip("-")
            tamt = tamt.lstrip("-")

        if not qty and uprice and ltot and abs(num(uprice) - num(ltot)) <= 0.01 and num(ltot) != 0:
            qty = "1"
        elif not uprice and qty == "1" and ltot:
            uprice = ltot
        elif not qty and not uprice and ltot:
            qty = "1"
            uprice = ltot

        li_taxes = []
        for lt in li.get("taxes") or []:
            if not isinstance(lt, dict):
                continue
            l_rate = clean_num_str(lt.get("tax_rate", "")).lstrip("-")
            l_amt = clean_num_str(lt.get("tax_amount", ""))
            if is_credit:
                l_amt = l_amt.lstrip("-")
            li_taxes.append({
                "tax_type": str(lt.get("tax_type") or "VAT").strip(),
                "tax_name": str(lt.get("tax_name") or "").strip(),
                "tax_rate": l_rate,
                "tax_amount": l_amt,
                "tax_type_code": str(lt.get("tax_type_code") or "").strip(),
            })

        item_type = str(li.get("item_type") or "GOODS").strip().upper()
        if item_type not in ("GOODS", "SERVICE", "FREIGHT", "TAX"):
            item_type = "GOODS"

        clean_lines.append({
            "description": str(li.get("description") or "").strip(),
            "item_type": item_type,
            "uom": str(li.get("uom") or "").strip(),
            "quantity": qty,
            "unit_price": uprice,
            "total": ltot,
            "discount": disc,
            "discount_percentage": disc_pct,
            "tax_rate": trate,
            "tax_amount": tamt,
            "taxes": li_taxes,
        })
    out["line_items"] = clean_lines

    meta = out.get("_meta")
    final_out = {}
    for k in SCHEMA_TOP_KEYS:
        if k in out:
            final_out[k] = out[k]
        elif k in ("taxes", "line_items"):
            final_out[k] = []
        elif k == "supplier":
            final_out[k] = {"name": "", "supplier_id": "", "address": "", "vat_id": ""}
        elif k == "buyer":
            final_out[k] = {"company_code": "", "business_unit_code": "", "location_code": ""}
        else:
            final_out[k] = ""
    if meta is not None:
        final_out["_meta"] = meta
    return final_out


def generate_diagnostic_signals(payable: dict) -> list[dict[str, str]]:
    """Evaluates a payable via erp.py mechanics and emits target-less factual signals."""
    signals: list[dict[str, str]] = []
    booked = erp_book(payable)["will_book_gross"]
    printed_gross = num(payable.get("gross_total"))

    sum_line_bases = 0.0
    sum_line_taxes = 0.0
    for idx, li in enumerate(payable.get("line_items") or []):
        base = _line_base(li)
        ltax = _line_taxes(li, base)
        sum_line_bases += base
        sum_line_taxes += ltax
        printed_ltot = num(li.get("total"))

        if printed_ltot != 0 and abs(base - printed_ltot) > 0.02:
            if num(li.get("discount_percentage")) > 0 and num(li.get("discount")) > 0:
                signals.append({
                    "field_path": f"line_items[{idx}]",
                    "kind": "LINE_DISCOUNT_MASKED",
                    "message": (
                        f"erp.py _line_base evaluates discount_percentage ({li.get('discount_percentage')}) "
                        f"and ignores discount ({li.get('discount')}) when both are set on line {idx} "
                        f"(derived base={base:.2f} vs line total={printed_ltot:.2f})."
                    ),
                })
            else:
                signals.append({
                    "field_path": f"line_items[{idx}]",
                    "kind": "LINE_EXTENSION_DIVERGENCE",
                    "message": (
                        f"erp.py derived _line_base={base:.2f} (line_tax={ltax:.2f}) from "
                        f"quantity={li.get('quantity')}, unit_price={li.get('unit_price')}, "
                        f"discount={li.get('discount')}, discount_percentage={li.get('discount_percentage')}, "
                        f"which differs from the line's stated total ({printed_ltot:.2f})."
                    ),
                })

    hdr_discount = abs(num(payable.get("discount_amount")))
    net_base = round2(sum_line_bases - hdr_discount)
    hdr_tax = _header_taxes(payable.get("taxes"), net_base)

    if sum_line_taxes != 0 and hdr_tax != 0:
        signals.append({
            "field_path": "taxes",
            "kind": "SIMULTANEOUS_LINE_AND_HEADER_TAX",
            "message": (
                f"erp.py sums both line_tax_total ({sum_line_taxes:.2f}) and header_tax ({hdr_tax:.2f}) "
                f"independently into the gross (combined tax={sum_line_taxes + hdr_tax:.2f})."
            ),
        })

    # Target-less component summary signal when ERP recompute does not reconcile
    if printed_gross > 0 and abs(booked - printed_gross) > 0.02:
        delta = round2(booked - printed_gross)
        other_charges = (
            num(payable.get("freight_charges"))
            + num(payable.get("insurance_charges"))
            + num(payable.get("extra_charges"))
            + num(payable.get("excise_duties"))
        )
        signals.append({
            "field_path": "payable",
            "kind": "UNRECONCILED_COMPONENT_SUM",
            "message": (
                f"ERP component recompute exhibits a net residual of {delta:+.2f} "
                f"(derived from sum_line_bases={sum_line_bases:.2f}, header_discount={hdr_discount:.2f}, "
                f"line_tax_total={sum_line_taxes:.2f}, header_tax={hdr_tax:.2f}, other_charges={other_charges:.2f})."
            ),
        })

    return signals