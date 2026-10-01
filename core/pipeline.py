import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_MODEL = None

# Lightweight offline normalization for common procurement terms.
# This is intentionally a keyword layer, not a claim of full machine translation.
LANGUAGE_TERMS = {
    "Hindi": {
        "केबल": "cable", "तार": "cable", "विद्युत": "electrical",
        "तांबा": "copper", "एल्युमिनियम": "aluminum", "पीवीसी": "pvc",
        "स्टील": "steel", "रबर": "rubber", "सीमेंट": "cement",
        "हेलमेट": "helmet", "पाइप": "pipe", "पंप": "pump",
        "पानी": "water", "टैंक": "tank", "भवन": "building",
        "निर्माण": "construction", "औद्योगिक": "industrial",
        "सुरक्षा": "safety", "परीक्षण": "testing", "जांच": "inspection",
        "इन्सुलेटेड": "insulated", "अग्निरोधी": "flame retardant",
        "जलरोधक": "waterproof", "संक्षारण": "corrosion",
    },
    "Telugu": {
        "కేబుల్": "cable", "తీగ": "cable", "విద్యుత్": "electrical",
        "రాగి": "copper", "అల్యూమినియం": "aluminum", "పివిసి": "pvc",
        "ఉక్కు": "steel", "రబ్బరు": "rubber", "సిమెంట్": "cement",
        "హెల్మెట్": "helmet", "పైపు": "pipe", "పంపు": "pump",
        "నీరు": "water", "ట్యాంక్": "tank", "భవనం": "building",
        "నిర్మాణం": "construction", "పారిశ్రామిక": "industrial",
        "భద్రత": "safety", "పరీక్ష": "testing", "తనిఖీ": "inspection",
        "ఇన్సులేటెడ్": "insulated", "అగ్ని నిరోధక": "flame retardant",
        "నీటి నిరోధక": "waterproof",
    },
    "Tamil": {
        "கேபிள்": "cable", "மின்சாரம்": "electrical", "செம்பு": "copper",
        "அலுமினியம்": "aluminum", "பிவிசி": "pvc", "எஃகு": "steel",
        "ரப்பர்": "rubber", "சிமெண்டு": "cement", "ஹெல்மெட்": "helmet",
        "குழாய்": "pipe", "பம்ப்": "pump", "தண்ணீர்": "water",
        "தொட்டி": "tank", "கட்டிடம்": "building", "கட்டுமானம்": "construction",
        "தொழில்துறை": "industrial", "பாதுகாப்பு": "safety",
        "சோதனை": "testing", "ஆய்வு": "inspection", "காப்பிடப்பட்ட": "insulated",
        "நீர்ப்புகா": "waterproof",
    },
    "Kannada": {
        "ಕೇಬಲ್": "cable", "ವಿದ್ಯುತ್": "electrical", "ತಾಮ್ರ": "copper",
        "ಅಲ್ಯೂಮಿನಿಯಂ": "aluminum", "ಪಿವಿಸಿ": "pvc", "ಉಕ್ಕು": "steel",
        "ರಬ್ಬರ್": "rubber", "ಸಿಮೆಂಟ್": "cement", "ಹೆಲ್ಮೆಟ್": "helmet",
        "ಪೈಪ್": "pipe", "ಪಂಪ್": "pump", "ನೀರು": "water",
        "ಟ್ಯಾಂಕ್": "tank", "ಕಟ್ಟಡ": "building", "ನಿರ್ಮಾಣ": "construction",
        "ಕೈಗಾರಿಕಾ": "industrial", "ಸುರಕ್ಷತೆ": "safety",
        "ಪರೀಕ್ಷೆ": "testing", "ತಪಾಸಣೆ": "inspection", "ನೀರಿನ ನಿರೋಧಕ": "waterproof",
    },
    "Malayalam": {
        "കേബിൾ": "cable", "വൈദ്യുതി": "electrical", "ചെമ്പ്": "copper",
        "അലുമിനിയം": "aluminum", "പിവിസി": "pvc", "സ്റ്റീൽ": "steel",
        "റബ്ബർ": "rubber", "സിമന്റ്": "cement", "ഹെൽമറ്റ്": "helmet",
        "പൈപ്പ്": "pipe", "പമ്പ്": "pump", "വെള്ളം": "water",
        "ടാങ്ക്": "tank", "കെട്ടിടം": "building", "നിർമ്മാണം": "construction",
        "വ്യാവസായിക": "industrial", "സുരക്ഷ": "safety",
        "പരിശോധന": "testing", "ഇൻസ്പെക്ഷൻ": "inspection", "വാട്ടർപ്രൂഫ്": "waterproof",
    },
    "Marathi": {
        "केबल": "cable", "वीज": "electrical", "तांबे": "copper",
        "अॅल्युमिनियम": "aluminum", "पीव्हीसी": "pvc", "स्टील": "steel",
        "रबर": "rubber", "सिमेंट": "cement", "हेल्मेट": "helmet",
        "पाईप": "pipe", "पंप": "pump", "पाणी": "water",
        "टाकी": "tank", "इमारत": "building", "बांधकाम": "construction",
        "औद्योगिक": "industrial", "सुरक्षा": "safety",
        "चाचणी": "testing", "तपासणी": "inspection", "जलरोधक": "waterproof",
    },
    "Bengali": {
        "কেবল": "cable", "বিদ্যুৎ": "electrical", "তামা": "copper",
        "অ্যালুমিনিয়াম": "aluminum", "পিভিসি": "pvc", "ইস্পাত": "steel",
        "রাবার": "rubber", "সিমেন্ট": "cement", "হেলমেট": "helmet",
        "পাইপ": "pipe", "পাম্প": "pump", "জল": "water",
        "ট্যাঙ্ক": "tank", "ভবন": "building", "নির্মাণ": "construction",
        "শিল্প": "industrial", "নিরাপত্তা": "safety",
        "পরীক্ষা": "testing", "পরিদর্শন": "inspection", "জলরোধী": "waterproof",
    },
    "Gujarati": {
        "કેબલ": "cable", "વીજ": "electrical", "તાંબુ": "copper",
        "એલ્યુમિનિયમ": "aluminum", "પીવીસી": "pvc", "સ્ટીલ": "steel",
        "રબર": "rubber", "સિમેન્ટ": "cement", "હેલ્મેટ": "helmet",
        "પાઇપ": "pipe", "પંપ": "pump", "પાણી": "water",
        "ટાંકી": "tank", "મકાન": "building", "બાંધકામ": "construction",
        "ઔદ્યોગિક": "industrial", "સલામતી": "safety",
        "પરીક્ષણ": "testing", "તપાસ": "inspection", "વોટરપ્રૂફ": "waterproof",
    },
    "Punjabi": {
        "ਕੇਬਲ": "cable", "ਬਿਜਲੀ": "electrical", "ਤਾਂਬਾ": "copper",
        "ਐਲੂਮੀਨੀਅਮ": "aluminum", "ਪੀਵੀਸੀ": "pvc", "ਸਟੀਲ": "steel",
        "ਰਬੜ": "rubber", "ਸੀਮੈਂਟ": "cement", "ਹੈਲਮੈਟ": "helmet",
        "ਪਾਈਪ": "pipe", "ਪੰਪ": "pump", "ਪਾਣੀ": "water",
        "ਟੈਂਕ": "tank", "ਇਮਾਰਤ": "building", "ਉਸਾਰੀ": "construction",
        "ਉਦਯੋਗਿਕ": "industrial", "ਸੁਰੱਖਿਆ": "safety",
        "ਟੈਸਟ": "testing", "ਜਾਂਚ": "inspection", "ਵਾਟਰਪ੍ਰੂਫ": "waterproof",
    },
    "Odia": {
        "କେବୁଲ": "cable", "ବିଦ୍ୟୁତ": "electrical", "ତମ୍ବା": "copper",
        "ଆଲୁମିନିୟମ": "aluminum", "ପିଭିସି": "pvc", "ଇସ୍ପାତ": "steel",
        "ରବର": "rubber", "ସିମେଣ୍ଟ": "cement", "ହେଲମେଟ": "helmet",
        "ପାଇପ": "pipe", "ପମ୍ପ": "pump", "ପାଣି": "water",
        "ଟାଙ୍କି": "tank", "କୋଠା": "building", "ନିର୍ମାଣ": "construction",
        "ଶିଳ୍ପ": "industrial", "ସୁରକ୍ଷା": "safety",
        "ପରୀକ୍ଷା": "testing", "ଯାଞ୍ଚ": "inspection", "ଜଳରୋଧୀ": "waterproof",
    },
    "Assamese": {
        "কেবল": "cable", "বিদ্যুৎ": "electrical", "তাম": "copper",
        "এলুমিনিয়াম": "aluminum", "পিভিচি": "pvc", "তীখা": "steel",
        "ৰবৰ": "rubber", "চিমেণ্ট": "cement", "হেলমেট": "helmet",
        "পাইপ": "pipe", "পাম্প": "pump", "পানী": "water",
        "টেংক": "tank", "ভৱন": "building", "নিৰ্মাণ": "construction",
        "ঔদ্যোগিক": "industrial", "সুৰক্ষা": "safety",
        "পৰীক্ষা": "testing", "পৰিদৰ্শন": "inspection", "জলৰোধী": "waterproof",
    },
}

def normalize_language_text(text, language="English"):
    if not text:
        return ""

    normalized = str(text)

    for source, target in LANGUAGE_TERMS.get(language, {}).items():
        normalized = normalized.replace(source, f" {target} ")

    # A few common mixed-language procurement words.
    for source, target in {
        "1.1 kv": "1.1 kV",
        "1.1KV": "1.1 kV",
        "kv": "kV",
    }.items():
        normalized = normalized.replace(source, target)

    return normalized

def _tokens(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9.-]*", text.lower()))

def extract_requirements(text, language="English"):
    # Normalize common terms before the existing English keyword extractor.
    t = normalize_language_text(text, language).lower()

    materials = []
    for x in [
        "copper", "aluminium", "aluminum", "pvc", "rubber", "steel",
        "stainless steel", "concrete", "polymer", "plastic", "glass", "cement"
    ]:
        if x in t:
            materials.append(x)

    properties = []
    for x in [
        "flame retardant", "fire resistant", "waterproof",
        "corrosion resistant", "impact resistant", "thermal",
        "pressure", "safety", "insulated", "insulation"
    ]:
        if x in t:
            properties.append(x)

    voltage = re.findall(r"\b\d+(?:\.\d+)?\s*(?:kv|v)\b", t)
    dimensions = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m)\b", t)

    tests = []
    for x in ["test", "testing", "inspection", "acceptance", "performance"]:
        if x in t:
            tests.append(x)

    products = [
        ("cable", "electrical cable"),
        ("helmet", "safety helmet"),
        ("water tank", "water storage tank"),
        ("pump", "water pump"),
        ("cement", "cement"),
        ("steel", "steel product"),
        ("transformer", "transformer"),
        ("switch", "electrical switch"),
        ("pipe", "pipe"),
        ("glove", "protective glove"),
    ]

    product = "Unknown product"
    for key, value in products:
        if key in t:
            product = value
            break

    application = "General procurement"
    for key, value in [
        ("building", "building installation"),
        ("industrial", "industrial use"),
        ("drinking water", "drinking water"),
        ("construction", "construction"),
        ("worker", "workplace safety"),
        ("electrical", "electrical installation"),
    ]:
        if key in t:
            application = value
            break

    return {
        "product": product,
        "materials": list(dict.fromkeys(materials)),
        "properties": list(dict.fromkeys(properties)),
        "voltage": voltage,
        "dimensions": dimensions,
        "tests": tests,
        "application": application,
        "raw": text[:10000],
        "normalized_input": t[:10000],
        "language": language,
    }

def _semantic_scores(query, docs):
    global _MODEL

    if SentenceTransformer:
        try:
            if _MODEL is None:
                _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            q = _MODEL.encode([query], normalize_embeddings=True)
            d = _MODEL.encode(docs, normalize_embeddings=True)
            return (q @ d.T)[0]
        except Exception:
            pass

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000
    )
    X = vec.fit_transform([query] + docs)
    return cosine_similarity(X[0:1], X[1:])[0]

def analyze_text(text, standards, language="English"):
    req = extract_requirements(text, language)

    query = " ".join([
        req["product"],
        " ".join(req["materials"]),
        " ".join(req["properties"]),
        " ".join(req["voltage"]),
        req["application"],
        req["normalized_input"][:3000],
    ])

    docs = (
        standards["title"].astype(str) + " " +
        standards["scope"].astype(str) + " " +
        standards["keywords"].astype(str) + " " +
        standards["product_categories"].astype(str) + " " +
        standards["applications"].astype(str)
    ).tolist()

    sem = _semantic_scores(query, docs)
    qtokens = _tokens(query)
    results = []

    for i, row in standards.iterrows():
        dtokens = _tokens(docs[i])
        lexical = len(qtokens & dtokens) / max(1, len(qtokens))

        categories = str(row["product_categories"]).lower()
        first_word = req["product"].lower().split()[0]

        product_match = (
            1.0
            if req["product"] == "Unknown product"
            or first_word in categories
            else 0.0
        )

        doc_lower = docs[i].lower()
        prop_hits = sum(
            1 for p in req["properties"] if p in doc_lower
        )
        mat_hits = sum(
            1 for m in req["materials"] if m in doc_lower
        )

        metadata = min(
            1.0,
            0.35 * product_match
            + 0.15 * min(1, prop_hits)
            + 0.15 * min(1, mat_hits)
            + 0.35 * float(str(row["status"]) == "Current")
        )

        raw = (
            0.45 * float(sem[i])
            + 0.25 * lexical
            + 0.30 * metadata
        )

        score = int(round(max(0, min(100, 50 + 50 * raw))))

        reasons = []
        if product_match:
            reasons.append(
                "Product category is compatible with the extracted product."
            )
        if mat_hits:
            reasons.append(
                "Material requirement appears in the standard metadata."
            )
        if prop_hits:
            reasons.append(
                "Requested performance/safety property is covered in the standard metadata."
            )
        if (
            req["application"]
            and req["application"].lower() in doc_lower
        ):
            reasons.append(
                "Application context matches the standard scope."
            )
        if str(row["status"]) == "Current":
            reasons.append("Dataset marks this standard as current.")

        if not reasons:
            reasons.append(
                "Semantic and lexical similarity identified this as a related candidate."
            )

        evidence = [
            f"Input language: {language}",
            f"Query attributes: {req['product']}, "
            f"{', '.join(req['materials']) or 'none'}",
            f"Matched scope/keywords: {str(row['scope'])[:180]}",
        ]

        results.append({
            "standard_id": row["standard_id"],
            "title": row["title"],
            "scope": row["scope"],
            "score": score,
            "status": row["status"],
            "classification": row["classification"],
            "sector": row["sector"],
            "reasons": reasons[:4],
            "evidence": evidence,
            "mandatory": row["mandatory"],
            "certification": row["certification"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:7]

    compliance = compliance_analysis(req, top)
    specification = generate_spec(req, top)

    return {
        "input": text,
        "language": language,
        "requirements": req,
        "recommendations": top,
        "compliance": compliance,
        "generated_specification": specification,
    }

def compliance_analysis(req, recs):
    checks = []

    product = req["product"] != "Unknown product"
    checks.append(
        ("Product identified", "Met" if product else "Missing")
    )
    checks.append(
        (
            "Applicable standard candidate identified",
            "Met" if recs else "Missing",
        )
    )
    checks.append(
        (
            "Material/property information",
            "Met" if req["materials"] or req["properties"] else "Warning",
        )
    )
    checks.append(
        (
            "Testing/inspection requirement",
            "Met" if req["tests"] else "Warning",
        )
    )
    checks.append(
        (
            "Application/use context",
            "Met"
            if req["application"] != "General procurement"
            else "Warning",
        )
    )
    checks.append(
        (
            "Certification/QCO evidence",
            "Met"
            if any(x["certification"] for x in recs[:3])
            else "Warning",
        )
    )

    items = [
        {"requirement": name, "status": status}
        for name, status in checks
    ]
    missing = [
        name for name, status in checks if status == "Missing"
    ]
    warnings = [
        name for name, status in checks if status == "Warning"
    ]
    coverage = round(
        100 * (
            sum(status == "Met" for _, status in checks)
            / len(checks)
        )
    )

    return {
        "items": items,
        "missing": missing,
        "warnings": warnings,
        "coverage": coverage,
    }

def generate_spec(req, recs):
    lines = [
        "PROCUREMENT-READY TECHNICAL SPECIFICATION (DRAFT)",
        "",
        f"1. Product: {req['product']}",
        f"2. Application: {req['application']}",
        f"3. Materials: {', '.join(req['materials']) or 'To be specified'}",
        f"4. Properties: {', '.join(req['properties']) or 'To be specified'}",
        f"5. Technical parameters: "
        f"{', '.join(req['voltage'] + req['dimensions']) or 'To be specified'}",
        "",
        "6. Applicable Indian Standards (subject to verification):",
    ]

    for r in recs[:3]:
        lines.append(
            f"- {r['standard_id']} — {r['title']} [{r['status']}]"
        )

    lines += [
        "",
        "7. Testing and inspection:",
        "- Supplier shall provide applicable test reports/certificates required by the referenced standards.",
        "- Purchaser shall verify conformity with the applicable clauses before acceptance.",
        "",
        "8. Certification/compliance:",
        "- Applicable certification, marking, QCO and regulatory requirements shall be verified against current authoritative notifications.",
        "",
        "9. Documentation:",
        "- Supplier shall submit product datasheet, test reports, certificates and traceability documents.",
        "",
        "NOTE: This AI-generated draft is a decision-support artifact. "
        "Verify all standard numbers, editions, clauses and regulatory "
        "requirements against authoritative BIS/government sources before "
        "issuing a tender.",
    ]

    return "\n".join(lines)
