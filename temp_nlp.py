

import random

modalities = ["X-ray", "MRI", "CT", "Ultrasound"]
institutions = ["St. Mary's Hospital", "Grand River Hospital", "True North Diagnostics"]
bodyparts = ["Head and Neck", "Chest", "Abdomen", "Upper Limbs", "Lower Limbs", "Other"]
clinicians = ["Dr. Grey", "Dr. Yang", "Dr. O'Malley", "Dr. Stevens", "Dr. Karev"]
dates = ["2022-01-15", "2021-09-22", "2021-04-04", "2020-01-01"]


def generate_random_tags():
    modality = random.choice(modalities)
    institution = random.choice(institutions)
    bodypart = random.choice(bodyparts)
    clinician = random.choice(clinicians)
    date = random.choice(dates)

    return [modality, bodypart, institution, clinician, date]
    # add these items to the database
