"""Tests for the maths tools (src/tools.py). No AI model needed."""

from tools import calculate_bmi, calculate_weight_based_dose, compute_dose, load_dosing_rules


def test_dosing_rules_are_parsed_from_guidelines():
    rules = load_dosing_rules()
    assert set(rules) == {"paracetamol", "ibuprofen", "amoxicillin"}
    para = rules["paracetamol"]
    assert (para.mg_per_kg, para.max_single_dose_mg, para.concentration_mg_per_ml) == (15, 1000, 24)


def test_paracetamol_dose_and_volume():
    result = calculate_weight_based_dose.invoke({"medication": "Paracetamol", "weight_kg": 18})
    assert "FINAL DOSE: 270.0 mg" in result
    assert "11.2 mL" in result  # 270 / 24 = 11.25
    assert "Sources: medication_dosing_reference.md" in result


def test_dose_is_capped_at_maximum():
    result = calculate_weight_based_dose.invoke({"medication": "ibuprofen", "weight_kg": 80})
    assert "Capped" in result
    assert "FINAL DOSE: 400.0 mg" in result


def test_unknown_medication_is_not_invented():
    result = calculate_weight_based_dose.invoke({"medication": "morphine", "weight_kg": 70})
    assert result.startswith("Information not found in AfyaPlus guidelines")


def test_dose_rejects_impossible_weight():
    result = calculate_weight_based_dose.invoke({"medication": "paracetamol", "weight_kg": 18000})
    assert result.startswith("Error:")


def test_dose_rejects_non_numeric_weight_without_crashing():
    result = calculate_weight_based_dose.func(medication="paracetamol", weight_kg="eighteen")
    assert result.startswith("Error:")


def test_compute_dose_pure_arithmetic():
    assert compute_dose(10, 15, 1000, 24) == (150, False, 6.25)
    assert compute_dose(100, 15, 1000, None) == (1000, True, None)


def test_bmi_normal():
    assert "22.9 (normal weight)" in calculate_bmi.invoke({"weight_kg": 70, "height_cm": 175})


def test_bmi_rejects_metres_instead_of_cm():
    assert calculate_bmi.invoke({"weight_kg": 70, "height_cm": 1.75}).startswith("Error:")


def test_tools_have_docstring_schemas():
    schema = calculate_weight_based_dose.args
    assert set(schema) == {"medication", "weight_kg"}
    assert "description" in schema["weight_kg"]
