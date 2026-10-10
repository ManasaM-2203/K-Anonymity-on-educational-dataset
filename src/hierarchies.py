"""
Domain Generalization Hierarchies for Educational Student Data.

Defines standard, explainable abstraction levels for numerical and categorical
quasi-identifiers commonly present in academic institutions.
"""

from typing import Any, Union
import math
import numpy as np
import pandas as pd


# 1. City / Geographical Generalization
# Level 0: Exact City
# Level 1: State / Geographic Zone
# Level 2: Broader Macro Region
CITY_HIERARCHY_LEVEL_1 = {
    "Bengaluru": "Karnataka (South)",
    "Mysuru": "Karnataka (South)",
    "Mangaluru": "Karnataka (Coastal)",
    "Hubballi": "Karnataka (North)",
    "Belagavi": "Karnataka (North)",
    "Pune": "Maharashtra (West)",
    "Hyderabad": "Telangana (South)",
    "Chennai": "Tamil Nadu (South)",
    "Coimbatore": "Tamil Nadu (South)"
}

CITY_HIERARCHY_LEVEL_2 = {
    "Bengaluru": "South Zone",
    "Mysuru": "South Zone",
    "Mangaluru": "South Zone",
    "Hubballi": "South Zone",
    "Belagavi": "South Zone",
    "Pune": "West Zone",
    "Hyderabad": "South Zone",
    "Chennai": "South Zone",
    "Coimbatore": "South Zone"
}


# 2. Department / Academic Faculty Generalization
# Level 0: Specific Department
# Level 1: Academic Faculty / Cluster
# Level 2: Broad Discipline (STEM / Non-STEM)
DEPARTMENT_HIERARCHY_LEVEL_1 = {
    "Computer Science": "Computing & IT",
    "Information Science": "Computing & IT",
    "Data Science & AI": "Computing & IT",
    "Electronics & Comm": "Electrical & Electronics",
    "Mechanical Engg": "Core Engineering",
    "Civil Engg": "Core Engineering",
    "Biotechnology": "Bio & Applied Sciences"
}

DEPARTMENT_HIERARCHY_LEVEL_2 = {
    "Computer Science": "Engineering & Technology",
    "Information Science": "Engineering & Technology",
    "Data Science & AI": "Engineering & Technology",
    "Electronics & Comm": "Engineering & Technology",
    "Mechanical Engg": "Engineering & Technology",
    "Civil Engg": "Engineering & Technology",
    "Biotechnology": "Applied Sciences"
}


# 3. Year of Study Generalization
YEAR_HIERARCHY_LEVEL_1 = {
    "1st Year": "Junior Cohort (Year 1-2)",
    "2nd Year": "Junior Cohort (Year 1-2)",
    "3rd Year": "Senior Cohort (Year 3-4)",
    "4th Year": "Senior Cohort (Year 3-4)"
}


def generalize_age(val: Union[int, float], level: int = 1) -> str:
    """
    Generalizes student age into explainable academic age intervals.
    Level 1: 3-year intervals (e.g., '18-20', '21-23', '24+')
    Level 2: 5-year intervals (e.g., '18-22', '23-27')
    """
    if pd.isna(val):
        return "Unknown"
    try:
        age = int(round(float(val)))
    except (ValueError, TypeError):
        return str(val)

    if level == 1:
        if age <= 20:
            return "18-20"
        elif age <= 23:
            return "21-23"
        elif age <= 26:
            return "24-26"
        else:
            return "27+"
    else:  # Level 2
        if age <= 22:
            return "18-22"
        elif age <= 26:
            return "23-26"
        else:
            return "27+"


def generalize_cgpa(val: Union[int, float], level: int = 1) -> str:
    """
    Generalizes numeric CGPA into performance grade brackets.
    Level 1: Standard CGPA ranges
      - [9.0 - 10.0] -> Outstanding
      - [7.5 - 8.99] -> First Class with Distinction
      - [6.0 - 7.49] -> First Class
      - [< 6.0]      -> Second Class / Pass
    """
    if pd.isna(val):
        return "Unknown"
    try:
        cgpa = float(val)
    except (ValueError, TypeError):
        return str(val)

    if level == 1:
        if cgpa >= 9.0:
            return "9.0 - 10.0 (Distinction+)"
        elif cgpa >= 7.5:
            return "7.5 - 8.9 (Distinction)"
        elif cgpa >= 6.0:
            return "6.0 - 7.4 (First Class)"
        else:
            return "< 6.0 (Pass Class)"
    else:  # Level 2
        if cgpa >= 7.5:
            return ">= 7.5 (High Merit)"
        else:
            return "< 7.5 (Standard)"


def generalize_city(city: str, level: int = 1) -> str:
    """Generalizes city name to state zone (Level 1) or macro region (Level 2)."""
    if not isinstance(city, str):
        return str(city)
    if level == 1:
        return CITY_HIERARCHY_LEVEL_1.get(city, f"{city} (Regional)")
    return CITY_HIERARCHY_LEVEL_2.get(city, "Other Region")


def generalize_department(dept: str, level: int = 1) -> str:
    """Generalizes specific academic department to faculty/cluster."""
    if not isinstance(dept, str):
        return str(dept)
    if level == 1:
        return DEPARTMENT_HIERARCHY_LEVEL_1.get(dept, "Engineering / Tech")
    return DEPARTMENT_HIERARCHY_LEVEL_2.get(dept, "Higher Education")


def generalize_year(year: str, level: int = 1) -> str:
    """Generalizes year of study to Junior / Senior cohort."""
    if not isinstance(year, str):
        return str(year)
    return YEAR_HIERARCHY_LEVEL_1.get(year, str(year))
