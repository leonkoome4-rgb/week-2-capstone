"""Tests for the maths tools (src/tools.py). No AI model needed."""

from tools import calculate_bmi, calculate_weight_based_dose


def test_paracetamol_dose_and_volume():
    result = calculate_weight_based_dose.invoke(
        {"weight_kg": 18, "dose_mg_per_kg": 15, "max_single_dose_mg": 1000,
         "concentration_mg_per_ml": 24}
    )
    assert "270.0 mg" in result
    assert "11.2 mL" in result  # 270 / 24 = 11.25


def test_dose_is_capped_at_maximum():
    result = calculate_weight_based_dose.invoke(
        {"weight_kg": 80, "dose_mg_per_kg": 15, "max_single_dose_mg": 1000}
    )
    assert "Capped" in result
    assert "FINAL DOSE: 1000.0 mg" in result


def test_dose_rejects_impossible_weight():
    result = calculate_weight_based_dose.invoke({"weight_kg": 18000, "dose_mg_per_kg": 15})
    assert result.startswith("Error:")


def test_dose_rejects_zero_concentration():
    result = calculate_weight_based_dose.invoke(
        {"weight_kg": 18, "dose_mg_per_kg": 15, "concentration_mg_per_ml": 0}
    )
    assert result.startswith("Error:")


def test_bmi_normal():
    assert "22.9 (normal weight)" in calculate_bmi.invoke({"weight_kg": 70, "height_cm": 175})


def test_bmi_rejects_metres_instead_of_cm():
    assert calculate_bmi.invoke({"weight_kg": 70, "height_cm": 1.75}).startswith("Error:")


def test_tools_have_docstring_schemas():
    schema = calculate_weight_based_dose.args
    assert "weight_kg" in schema and "description" in schema["weight_kg"]
