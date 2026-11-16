"""Deterministic maths tools for the AfyaPlus agent.

LLMs are unreliable at arithmetic, so every number the agent reports comes from these
functions. Each tool validates its inputs and returns an "Error: ..." string instead of
raising, so a bad call never crashes the agent loop - the model reads the error and can
ask the user for corrected values.
"""

from __future__ import annotations

from langchain_core.tools import tool

# Plausibility limits: reject values that are almost certainly typos (e.g. grams vs kg).
MIN_WEIGHT_KG, MAX_WEIGHT_KG = 0.5, 350.0
MIN_HEIGHT_CM, MAX_HEIGHT_CM = 30.0, 250.0
MAX_DOSE_MG_PER_KG = 100.0


def _check_range(name: str, value: float, low: float, high: float) -> None:
    if value is None:
        raise ValueError(f"{name} is required.")
    value = float(value)
    if value != value:  # NaN check
        raise ValueError(f"{name} must be a number.")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}.")


@tool(parse_docstring=True)
def calculate_weight_based_dose(
    weight_kg: float,
    dose_mg_per_kg: float,
    max_single_dose_mg: float | None = None,
    concentration_mg_per_ml: float | None = None,
) -> str:
    """Calculate a weight-based medication dose in mg, and optionally the liquid volume in mL.

    Only use dose values (mg/kg, maximum dose, concentration) that came from the
    AfyaPlus guidelines tool - never invent them.

    Args:
        weight_kg: Patient weight in kilograms (0.5 to 350).
        dose_mg_per_kg: Dose per kilogram from the guidelines, in mg/kg (greater than 0).
        max_single_dose_mg: Maximum single dose in mg from the guidelines; the result is capped at this value.
        concentration_mg_per_ml: Syrup or suspension strength in mg per mL, used to convert the dose to mL.
    """
    try:
        _check_range("weight_kg", weight_kg, MIN_WEIGHT_KG, MAX_WEIGHT_KG)
        _check_range("dose_mg_per_kg", dose_mg_per_kg, 0.001, MAX_DOSE_MG_PER_KG)
        if max_single_dose_mg is not None and float(max_single_dose_mg) <= 0:
            raise ValueError("max_single_dose_mg must be greater than 0.")
        if concentration_mg_per_ml is not None and float(concentration_mg_per_ml) <= 0:
            raise ValueError("concentration_mg_per_ml must be greater than 0.")

        raw_dose = float(weight_kg) * float(dose_mg_per_kg)
        dose = raw_dose
        lines = [f"Calculated dose: {weight_kg} kg x {dose_mg_per_kg} mg/kg = {raw_dose:.1f} mg."]

        if max_single_dose_mg is not None and raw_dose > float(max_single_dose_mg):
            dose = float(max_single_dose_mg)
            lines.append(f"Capped at the maximum single dose: {dose:.1f} mg.")

        if concentration_mg_per_ml is not None:
            volume = dose / float(concentration_mg_per_ml)
            lines.append(
                f"Volume: {dose:.1f} mg / {concentration_mg_per_ml} mg/mL = {volume:.1f} mL."
            )
        lines.append(f"FINAL DOSE: {dose:.1f} mg per dose.")
        return " ".join(lines)
    except (TypeError, ValueError) as exc:
        return f"Error: could not calculate dose - {exc}"
    except Exception as exc:  # defensive: never let a tool crash the agent loop
        return f"Error: unexpected problem in dose calculation ({type(exc).__name__})."


@tool(parse_docstring=True)
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """Calculate body mass index (BMI) and its triage band.

    Args:
        weight_kg: Weight in kilograms (0.5 to 350).
        height_cm: Height in centimetres (30 to 250).
    """
    try:
        _check_range("weight_kg", weight_kg, MIN_WEIGHT_KG, MAX_WEIGHT_KG)
        _check_range("height_cm", height_cm, MIN_HEIGHT_CM, MAX_HEIGHT_CM)

        height_m = float(height_cm) / 100
        bmi = float(weight_kg) / (height_m**2)
        if bmi < 18.5:
            band = "underweight"
        elif bmi < 25:
            band = "normal weight"
        elif bmi < 30:
            band = "overweight"
        else:
            band = "obese"
        return f"BMI = {weight_kg} kg / ({height_m:.2f} m)^2 = {bmi:.1f} ({band})."
    except (TypeError, ValueError) as exc:
        return f"Error: could not calculate BMI - {exc}"
    except Exception as exc:
        return f"Error: unexpected problem in BMI calculation ({type(exc).__name__})."


MATH_TOOLS = [calculate_weight_based_dose, calculate_bmi]
