"""Deterministic maths tools for the AfyaPlus agent.

LLMs are unreliable at arithmetic, so every number the agent reports comes from these
functions. Each tool validates its inputs and returns an "Error: ..." string instead of
raising, so a bad call never crashes the agent loop - the model reads the error and can
ask the user for corrected values.

Grounded dosing: the dose tool does NOT accept clinical values (mg/kg, maximum dose,
concentration) from the model. It reads them from the approved AfyaPlus dosing document,
so a small model cannot invent or mistype a dose. The model only supplies the medication
name and the patient's weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

DOSING_FILE = Path(__file__).resolve().parent.parent / "knowledge_manual" / "medication_dosing_reference.md"
NOT_FOUND = "Information not found in AfyaPlus guidelines"

# Plausibility limits: reject values that are almost certainly typos (e.g. grams vs kg).
MIN_WEIGHT_KG, MAX_WEIGHT_KG = 0.5, 350.0
MIN_HEIGHT_CM, MAX_HEIGHT_CM = 30.0, 250.0


@dataclass(frozen=True)
class DosingRule:
    medication: str
    mg_per_kg: float
    max_single_dose_mg: float | None
    concentration_mg_per_ml: float | None
    frequency: str | None


def _number(text: str) -> float:
    return float(text.replace(",", ""))


@lru_cache(maxsize=1)
def load_dosing_rules(path: Path = DOSING_FILE) -> dict[str, DosingRule]:
    """Parse '## N. <Drug> (...)' sections of the dosing reference into structured rules."""
    rules: dict[str, DosingRule] = {}
    sections = re.split(r"^##\s+\d+\.\s+", path.read_text(encoding="utf-8"), flags=re.M)[1:]
    for section in sections:
        name = section.splitlines()[0].split("(")[0].strip()
        dose = re.search(r"Dose:\s*([\d.,]+)\s*mg per kg", section)
        if not dose:
            continue  # not a medication section (e.g. BMI bands)
        max_dose = re.search(r"Maximum single dose:\s*([\d.,]+)\s*mg", section)
        conc = re.search(r"which is\s*([\d.,]+)\s*mg per mL", section)
        freq = re.search(r"Frequency:\s*(.+?)\.?$", section, flags=re.M)
        rules[name.lower()] = DosingRule(
            medication=name,
            mg_per_kg=_number(dose.group(1)),
            max_single_dose_mg=_number(max_dose.group(1)) if max_dose else None,
            concentration_mg_per_ml=_number(conc.group(1)) if conc else None,
            frequency=freq.group(1).strip() if freq else None,
        )
    return rules


def _check_range(name: str, value: float, low: float, high: float) -> float:
    if value is None:
        raise ValueError(f"{name} is required.")
    value = float(value)
    if value != value:  # NaN check
        raise ValueError(f"{name} must be a number.")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}.")
    return value


def compute_dose(
    weight_kg: float,
    mg_per_kg: float,
    max_single_dose_mg: float | None = None,
    concentration_mg_per_ml: float | None = None,
) -> tuple[float, bool, float | None]:
    """Pure arithmetic: returns (dose_mg, was_capped, volume_ml)."""
    raw = weight_kg * mg_per_kg
    capped = max_single_dose_mg is not None and raw > max_single_dose_mg
    dose = max_single_dose_mg if capped else raw
    volume = dose / concentration_mg_per_ml if concentration_mg_per_ml else None
    return dose, capped, volume


@tool(parse_docstring=True)
def calculate_weight_based_dose(medication: str, weight_kg: float) -> str:
    """Calculate a weight-based medication dose (mg) and syrup volume (mL) using the official
    AfyaPlus dosing guidelines. The guideline values are looked up automatically.

    Args:
        medication: Medication name, e.g. "paracetamol", "ibuprofen" or "amoxicillin".
        weight_kg: Patient weight in kilograms (0.5 to 350).
    """
    try:
        weight = _check_range("weight_kg", weight_kg, MIN_WEIGHT_KG, MAX_WEIGHT_KG)
        if not medication or not str(medication).strip():
            raise ValueError("medication name is required.")

        rules = load_dosing_rules()
        rule = rules.get(str(medication).strip().lower())
        if rule is None:
            return f"{NOT_FOUND} (no dosing rule for '{medication}')."

        dose, capped, volume = compute_dose(
            weight, rule.mg_per_kg, rule.max_single_dose_mg, rule.concentration_mg_per_ml
        )
        parts = [
            f"{rule.medication}: {weight:g} kg x {rule.mg_per_kg:g} mg/kg = {weight * rule.mg_per_kg:.1f} mg."
        ]
        if capped:
            parts.append(f"Capped at the maximum single dose of {rule.max_single_dose_mg:g} mg.")
        parts.append(f"FINAL DOSE: {dose:.1f} mg per dose.")
        if volume is not None:
            parts.append(
                f"Syrup volume: {dose:.1f} mg / {rule.concentration_mg_per_ml:g} mg per mL = {volume:.1f} mL."
            )
        if rule.frequency:
            parts.append(f"Frequency: {rule.frequency}.")
        parts.append(f"Sources: {DOSING_FILE.name}")
        return " ".join(parts)
    except (TypeError, ValueError) as exc:
        return f"Error: could not calculate dose - {exc}"
    except OSError:
        return "Error: the dosing guidelines file could not be read."
    except Exception as exc:  # defensive: never let a tool crash the agent loop
        return f"Error: unexpected problem in dose calculation ({type(exc).__name__})."


@tool(parse_docstring=True)
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """Calculate body mass index (BMI) and its AfyaPlus triage band.

    Args:
        weight_kg: Weight in kilograms (0.5 to 350).
        height_cm: Height in centimetres (30 to 250).
    """
    try:
        weight = _check_range("weight_kg", weight_kg, MIN_WEIGHT_KG, MAX_WEIGHT_KG)
        height = _check_range("height_cm", height_cm, MIN_HEIGHT_CM, MAX_HEIGHT_CM)

        height_m = height / 100
        bmi = weight / (height_m**2)
        if bmi < 18.5:
            band = "underweight"
        elif bmi < 25:
            band = "normal weight"
        elif bmi < 30:
            band = "overweight"
        else:
            band = "obese - refer to the chronic disease programme for screening"
        return (
            f"BMI = {weight:g} kg / ({height_m:.2f} m)^2 = {bmi:.1f} ({band}). "
            f"Sources: {DOSING_FILE.name}"
        )
    except (TypeError, ValueError) as exc:
        return f"Error: could not calculate BMI - {exc}"
    except Exception as exc:
        return f"Error: unexpected problem in BMI calculation ({type(exc).__name__})."


MATH_TOOLS = [calculate_weight_based_dose, calculate_bmi]
