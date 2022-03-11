from .model_api import ModelApi
import dateparser
import warnings
import re

class MinimalModel(ModelApi):
    def __init__(self):
        super().__init__()

    def train(self):
        """Really simple training: just put parameters in dict."""
        print('Training: Assigning Parameters')
        # Dates
        expr = '.*date(?!( of birth)).*(?<=:).*'
        self.parameters['date_regex'] = expr

        # Body parts
        body_parts = ['head', 'spine', 'chest', 'abdomen', 'pelivs', 'arm', 'leg']
        self.parameters['body_parts'] = body_parts

        # Imaging modalities
        modalities = ['MRI', 'M.R.I', 'XRAY', 'X-ray', 'X-RAY', 'X-Ray', 'CT', 'C.T.', 'XR']
        self.parameters['modalities'] = modalities

        # Doctors
        doctors = ['Dr. John Steele',
                   'Dr. John Martin',
                   'Dr. Ethel Shea',
                   'Dr. Lynn Lewis',
                   'Dr. Aubrey Higgins']
        self.parameters['dr_names'] = doctors

        # Imaging clinics
        clinics = ['UC Baby', 'KINGSTON MRI', 'HEALTHVIEW MEDICAL IMAGING',
                   'Central Alberta Medical Imaging Services LTD',
                   'MIC MEDICAL IMAGING - COLLEGE PLAZA']
        self.parameters['imaging_clinics'] = [c.lower() for c in clinics]

    def update_results(self):
        """
        Dates: Use regex to find possible date strings, use date parser
        Body parts: Check if report contains body parts. Take first one mentioned
        Imaging modalities: Search in report for string. Take first one mentioned
        Doctors: Search in report for string.
        Imaging clinics: search in report for string.
        """
        for key, text in self.input_data.items():
            results = {}
            # Search capitalized text

            for modality in self.parameters['modalities']:
                m = re.search(modality, text)
                if m is not None:
                    # Force to be in the right output format
                    modality = m.group().lower()
                    if 'm' in modality:
                        results['Modality'] = 'MRI'
                    elif 'c' in modality:
                        results['Modality'] = 'CT'
                    else:
                        results['Modality'] = 'X-RAY'
                    break

            for dr_name in self.parameters['dr_names']:
                m = re.search(dr_name, text)
                if m is not None:
                    results['Doctor Name'] = m.group()
                    break

            # Search lower case text
            text = text.lower()

            for clinic in self.parameters['imaging_clinics']:
                m = re.search(clinic, text)
                if m is not None:
                    results['Clinic Name'] = m.group()
                    break

            body_parts = [re.search(part, text) for part in self.parameters['body_parts']
                          if re.search(part, text) is not None]
            if body_parts:
                body_parts.sort(key=lambda x: x.span()[0])  # Sort by position in text
                results['Body Part'] = body_parts[0].group()  # Take first one by position
                # Replace arm and leg with upper and lower limbs
                if results['Body Part'] == 'arm':
                    results['Body Part'] = 'UPPER LIMB'
                elif results['Body Part'] == 'leg':
                    results['Body Part'] = 'LOWER LIMB'
                # Turn to upper case since that's what UI expects
                results['Body Part'] = results['Body Part'].upper()

            m = re.search(self.parameters['date_regex'], text)
            if m is not None:
                date = m.group().split(':')[-1].strip()
                # Supress warnings from dateparser
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results['Date Taken'] = dateparser.parse(date).strftime('%Y-%m-%d')

            # Put Nones where nothing was found
            [results.setdefault(k, None) for k in self.RESULT_KEYS]
            self.results.setdefault(key, results)
