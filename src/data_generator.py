"""
Synthetic Educational Dataset Generator for K-Anonymity Demonstration.

Generates realistic student records with controlled statistical distributions
to demonstrate quasi-identifier combinations, equivalence classes, and privacy risks.
"""

import os
import random
import numpy as np
import pandas as pd
from typing import Optional


# Synthetic First Names and Last Names for realistic name synthesis
FIRST_NAMES_FEMALE = [
    "Aaradhya", "Ananya", "Diya", "Isha", "Kavya", "Meera", "Neha", "Pooja",
    "Priya", "Rhea", "Ritu", "Sanvi", "Shreya", "Sneha", "Tanvi", "Vidya",
    "Deepika", "Aditi", "Divya", "Anushka", "Swati", "Rashmi", "Keerthi", "Nandini"
]

FIRST_NAMES_MALE = [
    "Aarav", "Aditya", "Akash", "Amit", "Arjun", "Dev", "Karan", "Manish",
    "Nikhil", "Pranav", "Rahul", "Rohan", "Sachin", "Siddharth", "Varun", "Vikram",
    "Abhishek", "Gautam", "Kunal", "Mayank", "Ravi", "Sanjay", "Suresh", "Tarun"
]

FIRST_NAMES_OTHER = [
    "Alex", "Arya", "Chris", "Jordan", "Kiran", "Morgan", "Robin", "Sam"
]

LAST_NAMES = [
    "Bhat", "Deshmukh", "Gowda", "Hegde", "Iyer", "Joshi", "Kamat", "Kulkarni",
    "Murthy", "Nair", "Patil", "Rao", "Reddy", "Sharma", "Shetty", "Singh",
    "Verma", "Pai", "Nayak", "Shenoy", "Menon", "Prabhu", "Acharya", "Chavan"
]

# Educational Domain Attributes and Probabilities
DEPARTMENTS = [
    "Computer Science",
    "Information Science",
    "Electronics & Comm",
    "Mechanical Engg",
    "Civil Engg",
    "Data Science & AI",
    "Biotechnology"
]
DEPT_WEIGHTS = [0.28, 0.18, 0.18, 0.12, 0.08, 0.10, 0.06]

YEARS_OF_STUDY = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
YEAR_WEIGHTS = [0.27, 0.26, 0.24, 0.23]

CITIES = [
    "Bengaluru",
    "Mysuru",
    "Mangaluru",
    "Hubballi",
    "Belagavi",
    "Pune",
    "Hyderabad",
    "Chennai"
]
CITY_WEIGHTS = [0.35, 0.18, 0.14, 0.09, 0.07, 0.07, 0.05, 0.05]

GENDERS = ["Female", "Male", "Other"]
GENDER_WEIGHTS = [0.47, 0.49, 0.04]


def _assign_age_by_year(year: str, rng: random.Random) -> int:
    """Assign a statistically realistic age based on the student's year of study."""
    if year == "1st Year":
        return rng.choices([18, 19, 20], weights=[0.70, 0.25, 0.05])[0]
    elif year == "2nd Year":
        return rng.choices([19, 20, 21], weights=[0.65, 0.30, 0.05])[0]
    elif year == "3rd Year":
        return rng.choices([20, 21, 22], weights=[0.60, 0.35, 0.05])[0]
    else:  # 4th Year
        return rng.choices([21, 22, 23, 24], weights=[0.55, 0.35, 0.08, 0.02])[0]


def _determine_academic_performance(cgpa: float, rng: random.Random) -> str:
    """
    Determine categorical academic performance with slight noise to mimic real evaluation.
    This serves as a confidential/sensitive educational attribute.
    """
    if cgpa >= 8.5:
        return "Excellent"
    elif cgpa >= 7.0:
        return "Good"
    elif cgpa >= 5.5:
        return "Average"
    else:
        return "Needs Improvement"


def generate_student_dataset(n_records: int = 750, random_seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Generates a synthetic educational dataset with realistic statistical properties.

    Parameters:
        n_records (int): Total number of student records to generate (default: 750).
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic student records dataframe.
    """
    rng = random.Random(random_seed)
    np_rng = np.random.default_rng(random_seed)

    records = []

    # CGPA sampled from a realistic normal distribution truncated between [4.00, 10.00]
    cgpa_raw = np_rng.normal(loc=7.65, scale=1.15, size=n_records)
    cgpa_clipped = np.clip(cgpa_raw, 4.20, 9.95)

    for i in range(n_records):
        student_num = i + 1
        student_id = f"STU{2024000 + student_num:07d}"

        gender = rng.choices(GENDERS, weights=GENDER_WEIGHTS)[0]
        if gender == "Female":
            first_name = rng.choice(FIRST_NAMES_FEMALE)
        elif gender == "Male":
            first_name = rng.choice(FIRST_NAMES_MALE)
        else:
            first_name = rng.choice(FIRST_NAMES_OTHER)

        last_name = rng.choice(LAST_NAMES)
        full_name = f"{first_name} {last_name}"

        year = rng.choices(YEARS_OF_STUDY, weights=YEAR_WEIGHTS)[0]
        age = _assign_age_by_year(year, rng)
        department = rng.choices(DEPARTMENTS, weights=DEPT_WEIGHTS)[0]
        city = rng.choices(CITIES, weights=CITY_WEIGHTS)[0]
        cgpa = round(float(cgpa_clipped[i]), 2)
        academic_performance = _determine_academic_performance(cgpa, rng)

        records.append({
            "Student_ID": student_id,
            "Name": full_name,
            "Age": age,
            "Gender": gender,
            "Department": department,
            "Year": year,
            "City": city,
            "CGPA": cgpa,
            "Academic_Performance": academic_performance
        })

    df = pd.DataFrame(records)
    return df


def save_dataset_to_csv(df: pd.DataFrame, output_path: str = "data/students_dataset.csv") -> str:
    """Saves the generated dataframe to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    # Generate and save default dataset
    dataset = generate_student_dataset(n_records=750, random_seed=42)
    saved_path = save_dataset_to_csv(dataset)
    print(f"Generated {len(dataset)} synthetic student records.")
    print(f"Saved to: {saved_path}")
    print("\nDataset Preview:")
    print(dataset.head())
    print("\nAttribute Value Counts:")
    print("Departments:\n", dataset["Department"].value_counts())
    print("\nAcademic Performance (Sensitive Attribute):\n", dataset["Academic_Performance"].value_counts())
