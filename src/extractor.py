"""src/extractor.py — Principle-driven Vision-LLM solver with tiered intelligence routing."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.reconciler import generate_diagnostic_signals, sanitize_payable_schema
from src.resolver import MasterDataResolver


def _build_static_system_prompt() -> str:
    """Builds a static system prefix containing the ERP contract and universal accounting principles."""
    schema_text = ""
    if os.path.isfile("AUTODRAFT_SCHEMA.md"):
        with open("AUTODRAFT_SCHEMA.md", encoding="utf-8") as fh:
            schema_text = fh.read()

    return f"""You are an expert Accounts Payable specialist. Given a supplier PDF packet, determine what each page is, decline non-payable documents, and decompose each bookable payable into the raw components required by the ERP.

=== THE ERP RECOMPUTE CONTRACT (`erp.py`) ===
- Line net base: `sub_total = quantity * unit_price` (where `unit_price` is net / tax-exclusive).
  If `discount_percentage > 0`: `base = round2(sub_total * (1 - abs(discount_percentage)/100))`
  Else if `discount > 0` and `quantity != 0`: `base = round2((unit_price - abs(discount)/abs(quantity)) * quantity)`
  Else: `base = round2(sub_total)`
- Line taxes: for each entry in `line_items[].taxes[]` (or line `tax_rate`/`tax_amount`), if `tax_amount == 0` and `tax_rate > 0`, `amt = round2(base * tax_rate / 100)`; otherwise `amt = tax_amount`.
- Header net base: `net_base = sum(line_base) - abs(discount_amount)`
- Header taxes: for each entry in `taxes[]`, if `tax_amount == 0` and `tax_rate > 0`, `amt = round2(net_base * tax_rate / 100)`; otherwise `amt = tax_amount`.
- Other charges: `other_charges = freight_charges + insurance_charges + extra_charges + excise_duties`.
- Booked gross: `will_book_gross = round2(net_base + sum(line_taxes) + sum(header_taxes) + other_charges)`.

=== GOVERNING PRINCIPLES ===
1. Grounding: Every monetary figure, quantity, and rate emitted must appear on the document. Never invent a balancing plug figure or ungrounded rate.
2. Structural fidelity: Represent each tax, discount, and charge once at the level (line vs. header) where the document applies it. Keep raw components decomposed.
3. Packet segmentation: A PDF may contain multiple documents, duplicate copies, remittance slips, supporting customs/meter breakdowns, or non-payable records (quotes/estimates, dunning reminders, delivery notes, internal forms). Emit one payable per genuine commercial invoice or credit memo, and place non-payables in `declined[]`.

=== SCHEMA SPECIFICATION ===
{schema_text}

=== OUTPUT JSON ENVELOPE ===
Return a JSON object with keys `"payables"` (list of payable objects; each may include an optional `"_meta"` object with `"supplier_country"`, `"supplier_iban"`, `"supplier_email"`, `"buyer_name"`, `"buyer_address"`, `"payment_term_text"`) and `"declined"` (list of `{{"doc_type": "...", "reason": "..."}}`)."""


STATIC_SYSTEM_PROMPT = _build_static_system_prompt()


class DocumentSolver:
    """Manages tiered model routing, prompt prefix reuse, and signal-driven reconciliation."""

    def __init__(self, resolver: MasterDataResolver) -> None:
        load_dotenv(override=True)
        raw_keys = os.environ.get("GEMINI_API_KEY", "")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not self.api_keys:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file or environment.")
        self.key_index = 0
        self.client = genai.Client(api_key=self.api_keys[self.key_index])
        # Tiered routing: fast model for initial extraction; reasoning model for reconciliation repairs
        self.fast_model = os.environ.get("GEMINI_FAST_MODEL", "gemini-2.5-flash")
        self.solver_model = os.environ.get("GEMINI_SOLVER_MODEL", "gemini-2.5-flash")
        self.resolver = resolver
        self.max_turns = 3

    def _rotate_key(self) -> bool:
        if self.key_index + 1 < len(self.api_keys):
            self.key_index += 1
            print(f"\n[Quota reached] Rotating to API key #{self.key_index + 1}...", end=" ", flush=True)
            self.client = genai.Client(api_key=self.api_keys[self.key_index])
            return True
        return False

    def _call_llm(self, contents: list[Any], is_repair_turn: bool = False) -> dict:
        """Routes initial reads to fast config and repair turns to deeper reasoning."""
        model = self.solver_model if is_repair_turn else self.fast_model
        thinking_budget = 2048 if is_repair_turn else 0
        config = types.GenerateContentConfig(
            system_instruction=STATIC_SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        while True:
            for attempt in range(3):
                try:
                    resp = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    text = (resp.text or "{}").strip()
                    if text.startswith("```"):
                        text = text.strip("`").removeprefix("json").strip()
                    return json.loads(text)
                except Exception as exc:
                    err_str = str(exc)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        if "PerDay" in err_str or attempt == 2:
                            if self._rotate_key():
                                break
                        time.sleep(6 * (attempt + 1))
                    else:
                        if attempt == 2:
                            raise
                        time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError("All provided GEMINI_API_KEYs have exhausted their quota.")

    def solve_pdf(self, pdf_path: str) -> dict:
        """Executes initial extraction followed by signal-driven ERP reconciliation."""
        filename = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()

        initial_task = (
            f"File: `{filename}`.\n"
            "Segment the PDF, decline non-payables, and extract bookable payables into the required JSON format."
        )

        contents: list[Any] = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    types.Part.from_text(text=initial_task),
                ],
            )
        ]

        draft = self._call_llm(contents, is_repair_turn=False)

        for _turn in range(1, self.max_turns):
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

            if not all_signals:
                break

            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=json.dumps(draft))],
                )
            )

            feedback_prompt = (
                "ERP diagnostic observations on the current draft:\n"
                f"{json.dumps(all_signals, indent=2)}\n\n"
                "Re-examine the PDF and revise the draft if any printed component, tax placement, discount, "
                "or sign was mis-mapped. Do not invent ungrounded numbers; if the document itself is incomplete, "
                "keep the grounded values as printed."
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=feedback_prompt)],
                )
            )

            new_draft = self._call_llm(contents, is_repair_turn=True)
            if json.dumps(new_draft, sort_keys=True) == json.dumps(draft, sort_keys=True):
                break
            draft = new_draft

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