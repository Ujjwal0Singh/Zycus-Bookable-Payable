"""src/extractor.py — Vision-LLM document solver with prompt-cache-friendly prefix and flowing multi-turn loop."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from erp import erp_book, num
from src.reconciler import generate_diagnostic_signals, sanitize_payable_schema
from src.resolver import MasterDataResolver

load_dotenv()


def _build_static_system_prompt() -> str:
    """Builds a completely static, cacheable system instruction prefix."""
    schema_text = ""
    if os.path.isfile("AUTODRAFT_SCHEMA.md"):
        with open("AUTODRAFT_SCHEMA.md", encoding="utf-8") as fh:
            schema_text = fh.read()

    return f"""You are an expert Accounts Payable reasoning agent. Your job is to examine a supplier PDF document, determine what each page represents, decline non-payables, and decompose every bookable payable into raw accounting components that reconstruct the document's true amount owed when run through `erp.py`.

=== THE ALGEBRAIC CONTRACT (HOW `erp.py` COMPUTES GROSS) ===
1. For each line in `line_items[]`:
   - `sub_total = quantity * unit_price` (where `unit_price` MUST be NET / tax-exclusive).
   - If `discount_percentage > 0`: `line_base = round2(sub_total - (sub_total * abs(discount_percentage) / 100))`
   - Else if `discount > 0` and `quantity != 0`: `line_base = round2((unit_price - abs(discount)/abs(quantity)) * quantity)`
   - Else: `line_base = round2(sub_total)`
   - Line taxes (`line_items[].taxes[]` or `tax_rate`/`tax_amount`): for each tax, if `tax_amount` is empty/0 and `tax_rate > 0`, `amt = round2(line_base * tax_rate / 100)`; otherwise `amt = tax_amount`.
2. At the header:
   - `net_base = sum(line_base) - abs(discount_amount)`
   - Header taxes (`taxes[]`): for each tax, if `tax_amount` is empty/0 and `tax_rate > 0`, `amt = round2(net_base * tax_rate / 100)`; otherwise `amt = tax_amount`.
   - `other_charges = freight_charges + insurance_charges + extra_charges + excise_duties` (added AFTER `net_base`, untaxed by header `tax_rate`).
   - `will_book_gross = round2(net_base + sum(line_taxes) + sum(header_taxes) + other_charges)`

=== CORE PRINCIPLES & INVARIANTS ===
1. GROUNDING (NO PLUGS): Every monetary number, quantity, rate, or discount you emit MUST literally appear on the document. Never invent a balancing plug line or fake a rate to force `will_book_gross` to match `gross_total`. If a document is missing lines on the page (e.g., skipped item numbers) or only prints tax-inclusive prices without enough page facts to decompose net prices deterministically, emit only the grounded values actually printed on the page.
2. STRUCTURAL FAITHFULNESS:
   - Place line-level taxes on the line (`line_items[].taxes[]`) and header-level taxes at the header (`taxes[]`). Never put the same tax in both places (which double-counts tax). When a bottom tax table is merely a summary of per-line taxes (or when per-line rounding of a single rate differs by 1 cent from the document's tax summary), ensure each tax is represented once at its true calculation base and supply the printed `tax_amount` when compound levies (e.g. Ghana NHIL/GETFund/COVID + 15% VAT on levy-inclusive subtotal) or grouped tax rounding apply.
   - Do not extract subtotals, carried-forward rows ("Overført", "Subtotal"), payment remittance tear-off slips, "Duplicado"/duplicate copy pages, or supporting customs/shipment specification pages as separate payables or extra line items.
   - For `CREDIT_MEMO`, set `invoice_type: "CREDIT_MEMO"` and emit all line/tax/total magnitudes as POSITIVE numbers.
   - For withholding taxes (e.g., WHT 3% that reduces the payment due), set `tax_amount` as a NEGATIVE number (e.g. `"-235.44"`), and set `gross_total` to the net payable amount after withholding if that is what the document states is owed.
   - Convert Buddhist Era years (e.g., 2569 BE -> 2026 CE) to ISO `YYYY-MM-DD`.
   - Decline non-payables (e.g. dunning letters / `Mahnung`, estimates / quotes, delivery notes, waybills, internal sponsorship approval forms) in `declined[]`.

=== SCHEMA SPECIFICATION ===
{schema_text}

=== REQUIRED JSON OUTPUT FORMAT ===
Always return a single JSON object with this shape:
{{
  "payables": [
    {{
      "invoice_number": "...",
      "invoice_date": "YYYY-MM-DD",
      "due_date": "YYYY-MM-DD",
      "invoice_type": "INVOICE" | "CREDIT_MEMO",
      "currency": "EUR",
      "supplier": {{ "name": "...", "supplier_id": "", "address": "...", "vat_id": "..." }},
      "buyer": {{ "company_code": "", "business_unit_code": "", "location_code": "" }},
      "payment_term_id": "",
      "po_number": "...",
      "po_id": "",
      "gross_total": "...",
      "subtotal": "...",
      "total_tax_amount": "...",
      "discount_amount": "",
      "freight_charges": "",
      "insurance_charges": "",
      "extra_charges": "",
      "excise_duties": "",
      "taxes": [ {{ "tax_type": "...", "tax_name": "...", "tax_rate": "...", "tax_amount": "...", "tax_type_code": "" }} ],
      "line_items": [
        {{
          "description": "...",
          "item_type": "GOODS" | "SERVICE" | "FREIGHT" | "TAX",
          "uom": "...",
          "quantity": "...",
          "unit_price": "...",
          "total": "...",
          "discount": "",
          "discount_percentage": "",
          "tax_rate": "",
          "tax_amount": "",
          "taxes": []
        }}
      ],
      "_meta": {{
        "supplier_country": "2-letter ISO country code of supplier",
        "supplier_iban": "printed IBAN if any",
        "supplier_email": "printed supplier email if any",
        "buyer_name": "printed buyer/customer legal name",
        "buyer_address": "printed buyer address",
        "payment_term_text": "printed payment terms phrase"
      }}
    }}
  ],
  "declined": [
    {{ "doc_type": "...", "reason": "..." }}
  ]
}}"""


STATIC_SYSTEM_PROMPT = _build_static_system_prompt()


class DocumentSolver:
    """Manages the Gemini client, cached system prefix, key rotation, and multi-turn reconciliation loop."""

    def __init__(self, resolver: MasterDataResolver) -> None:
        load_dotenv(override=True)
        raw_keys = os.environ.get("GEMINI_API_KEY", "")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not self.api_keys:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file or environment.")
        self.key_index = 0
        self.client = genai.Client(api_key=self.api_keys[self.key_index])
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.resolver = resolver
        self.max_turns = 3
    
    def _rotate_key(self) -> bool:
        """Switches to the next API key in GEMINI_API_KEY if available."""
        if self.key_index + 1 < len(self.api_keys):
            self.key_index += 1
            print(f"\n[Quota reached] Rotating to API key #{self.key_index + 1}...", end=" ", flush=True)
            self.client = genai.Client(api_key=self.api_keys[self.key_index])
            return True
        return False

    def _call_llm(self, contents: list[Any]) -> dict:
        """Calls Gemini with a static cacheable system prefix and JSON response enforcement."""
        config = types.GenerateContentConfig(
            system_instruction=STATIC_SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
        )
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                text = (resp.text or "{}").strip()
                if text.startswith("```"):
                    text = text.strip("`").removeprefix("json").strip()
                return json.loads(text)
            except Exception as exc:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        return {"payables": [], "declined": []}

    def solve_pdf(self, pdf_path: str) -> dict:
        """Runs the multi-turn extraction and ERP-signal reconciliation loop over one PDF."""
        filename = os.path.basename(pdf_path)
        uploaded_file = self.client.files.upload(file=pdf_path)

        try:
            initial_task = (
                f"Document file: `{filename}`.\n"
                "1. Inspect all pages of the attached PDF.\n"
                "2. Segment the packet into distinct documents. Ignore duplicate copy pages ('Duplicado', 'Extra Copy'), "
                "remittance tear-off slips, and supporting backup attachments that merely detail the main invoice.\n"
                "3. Put any non-payable documents into `declined[]` and extract all bookable payables into `payables[]` "
                "strictly grounded to the numbers printed on the document."
            )

            # Flowing multi-turn context: PDF at the top, active task at the bottom
            contents: list[Any] = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="application/pdf"),
                        types.Part.from_text(text=initial_task),
                    ],
                )
            ]

            draft = self._call_llm(contents)

            # Iterative Solver Loop driven by deterministic ERP signals
            for turn in range(1, self.max_turns):
                raw_payables = draft.get("payables") or []
                if not raw_payables:
                    break

                all_signals: list[dict] = []
                for p_idx, p in enumerate(raw_payables):
                    clean_p = sanitize_payable_schema(p)
                    sigs = generate_diagnostic_signals(clean_p)
                    for s in sigs:
                        s["payable_index"] = p_idx
                        all_signals.append(s)

                # Terminating loop immediately if all payables reconcile cleanly with zero diagnostic signals
                if not all_signals:
                    break

                # Appending model's previous output and feeding back factual signals at the bottom of context
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=json.dumps(draft))],
                    )
                )

                feedback_prompt = (
                    "=== ERP RECONCILIATION SIGNALS ===\n"
                    f"{json.dumps(all_signals, indent=2)}\n\n"
                    "Review the signals above against the attached PDF:\n"
                    "- If a signal indicates double-counted taxes (line + header), duplicate discounts, a subtotal/carried-forward "
                    "line included in `line_items`, an uncaptured printed charge/excise/deposit, a withholding tax sign, or a "
                    "compound/rounded tax where the printed `tax_amount` on the page should be used, update the JSON using ONLY "
                    "numbers printed on the document.\n"
                    "- CRITICAL: If the discrepancy is caused by missing rows on the document page itself (e.g. skipped item numbers) "
                    "or prices that cannot be decomposed using numbers printed on the page, DO NOT invent a plug number or ungrounded "
                    "rate—keep the grounded extraction as-is.\n"
                    "Return the complete updated JSON object."
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=feedback_prompt)],
                    )
                )

                new_draft = self._call_llm(contents)
                # Stop if the model made no structural changes (converged / irreducible document)
                if json.dumps(new_draft, sort_keys=True) == json.dumps(draft, sort_keys=True):
                    break
                draft = new_draft

            # Finalize: sanitize schema and resolve master-data codes
            final_payables = []
            for p in draft.get("payables") or []:
                clean_p = sanitize_payable_schema(p)
                enriched_p = self.resolver.enrich_payable(clean_p)
                final_payables.append(enriched_p)

            final_declined = []
            for d in draft.get("declined") or []:
                if isinstance(d, dict):
                    final_declined.append({
                        "doc_type": str(d.get("doc_type") or "NON_PAYABLE").strip(),
                        "reason": str(d.get("reason") or "Not a bookable payable document").strip(),
                    })

            return {
                "file": filename,
                "payables": final_payables,
                "declined": final_declined,
            }
        finally:
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass