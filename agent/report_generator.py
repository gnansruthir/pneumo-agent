import os
import json
import google.generativeai as genai

# Attempt to configure Gemini if key is in environment
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ReportGeneratorAgent:
    def __init__(self, guidelines_path=None):
        self.guidelines = {}
        if guidelines_path and os.path.exists(guidelines_path):
            try:
                with open(guidelines_path, 'r') as f:
                    self.guidelines = json.load(f)
            except Exception as e:
                print(f"Failed to load guidelines: {e}")

    def _get_rag_context(self, detected_findings):
        """
        Retrieves matching guidelines context for the detected findings.
        """
        context = []
        for finding in detected_findings:
            if finding in self.guidelines:
                info = self.guidelines[finding]
                context.append(
                    f"Finding: {finding}\n"
                    f"Definition: {info['definition']}\n"
                    f"Guideline: {info['who_recommendation']}\n"
                    f"Triage: {info['triage']}"
                )
        return "\n\n".join(context)

    def _generate_mock_report(self, detected_findings, triage, probabilities):
        """
        Fallback mock report generator in case LLM API keys are not provided.
        Generates structured clinical reports matching realistic outputs.
        """
        findings_str = ", ".join(detected_findings) if detected_findings else "No abnormal findings"
        
        # Build individual clinical details
        clinical_details = []
        for f in detected_findings:
            prob = probabilities.get(f, {}).get("probability", 0.0)
            guideline = self.guidelines.get(f, {}).get("who_recommendation", "Monitor clinically.")
            clinical_details.append(
                f"- **{f}** (probability: {prob:.2f}): Features suggest {f.lower()}. "
                f"Recommendations: {guideline}"
            )
            
        details_block = "\n".join(clinical_details) if clinical_details else "- No active cardiopulmonary disease or acute findings detected."
        
        clinical_report = f"""# CLINICAL RADIOLOGY REPORT
**Triage Category**: {triage.upper()}
**Indication**: Screening Chest X-Ray

## FINDINGS
{details_block}

## IMPRESSION
1. Image analysis is positive for: {findings_str}.
2. Triage priority is determined as **{triage}** based on findings severity and WHO clinical pathways.
"""

        english_summary = f"""# Patient Summary (English)
The chest X-ray analysis suggests the following:
- Detected conditions: {findings_str}
- Triage: {triage}

**What this means**:
{"Some areas of concern were detected on your chest X-ray. It is recommended to schedule a follow-up with your physician to evaluate these findings." if detected_findings else "Your chest X-ray appears clear of major acute conditions. Maintain standard health protocols."}
"""

        # Simple localized translations for the mock to feel premium
        hindi_findings = []
        for f in detected_findings:
            if f == "Tuberculosis": hindi_findings.append("तपेदिक (TB)")
            elif f == "Pneumonia": hindi_findings.append("निमोनिया (Pneumonia)")
            elif f == "Cardiomegaly": hindi_findings.append("बढ़ा हुआ हृदय (Cardiomegaly)")
            else: hindi_findings.append(f"{f}")
            
        hindi_findings_str = ", ".join(hindi_findings) if hindi_findings else "कोई असामान्य लक्षण नहीं"
        hindi_triage = "तत्काल (Urgent)" if triage == "Urgent" else "अनुवर्ती (Follow-up)" if triage == "Follow-up" else "स्पष्ट (Clear)"

        hindi_summary = f"""# मरीज सारांश (Hindi)
आपके छाती के एक्स-रे (Chest X-Ray) विश्लेषण के परिणाम:
- पाई गई स्थिति: {hindi_findings_str}
- प्राथमिकता: {hindi_triage}

**इसका क्या अर्थ है**:
{"आपके एक्स-रे में कुछ चिंताजनक लक्षण दिखाई दिए हैं। कृपया आगे की जांच और डॉक्टर से संपर्क करने के लिए जल्द अपॉइंटमेंट लें।" if detected_findings else "आपका एक्स-रे सामान्य है। किसी गंभीर बीमारी के लक्षण नहीं मिले हैं।"}
"""

        return {
            "clinical_report": clinical_report.strip(),
            "patient_summary_en": english_summary.strip(),
            "patient_summary_hi": hindi_summary.strip()
        }

    def generate_report(self, result):
        """
        Main interface to generate clinical and patient reports.
        """
        detected_findings = result.get("detected_findings", [])
        triage = result.get("triage", "Clear")
        probabilities = result.get("predictions", {})

        rag_context = self._get_rag_context(detected_findings)
        
        # Check if Gemini API key is active
        if os.getenv("GEMINI_API_KEY"):
            try:
                # Setup Gemini model
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"""
                You are a professional chest radiology AI assistant. Generate a chest X-ray clinical report and summaries.
                
                CLASSIFICATION RESULTS:
                Detected Findings: {detected_findings}
                Triage Priority: {triage}
                Finding Probabilities: {json.dumps(probabilities)}
                
                CLINICAL RAG CONTEXT (WHO guidelines & definitions):
                {rag_context}
                
                OUTPUT FORMAT:
                You must return a valid JSON object with exactly three string fields:
                1. "clinical_report": A markdown-formatted detailed radiology report (including findings, impression, and recommendations).
                2. "patient_summary_en": A patient-friendly markdown summary of the findings and next steps in English.
                3. "patient_summary_hi": A patient-friendly markdown summary of the findings and next steps translated to Hindi.
                
                Do not include backticks, markdown code blocks like ```json or any other text around the JSON. Output raw JSON only.
                """
                
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Strip out formatting if model returns it inside a markdown block
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                report_data = json.loads(response_text)
                return {
                    "clinical_report": report_data.get("clinical_report"),
                    "patient_summary_en": report_data.get("patient_summary_en"),
                    "patient_summary_hi": report_data.get("patient_summary_hi"),
                    "source": "Gemini LLM"
                }
            except Exception as e:
                print(f"Error using Gemini LLM: {e}. Falling back to mock generator.")
                
        # Return mock reports if no LLM active or failed
        report = self._generate_mock_report(detected_findings, triage, probabilities)
        report["source"] = "Clinical Template System"
        return report
