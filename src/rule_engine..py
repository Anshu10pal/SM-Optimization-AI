import streamlit as st
import pandas as pd

# -------------------------------
# LOAD SOP RULES
# -------------------------------
RULES_FILE = r"D:\ML\SM Modeling\Data\Master\SOP_rules.xlsx"
rules = pd.read_excel(RULES_FILE)
rules = rules.dropna(subset=["rule_id", "keywords"])

# -------------------------------
# CONFIG
# -------------------------------
CONFIDENCE_THRESHOLD = 0.85

EXCLUSION_KEYWORDS = [
    "proactive", "auto restore", "service restored",
    "tlos", "monitoring", "cleared automatically"
]

EXTERNAL_NETWORK_KEYWORDS = [
    "cabinet", "toby", "asn", "pn", "sn", "dp",
    "pole", "pit", "chamber", "road", "street"
]

CUSTOMER_PREMISES_KEYWORDS = [
    "ont", "wallbox", "patch lead", "router",
    "inside", "customer premises"
]

# -------------------------------
# DECISION FUNCTION
# -------------------------------
def decide_incident(
    current_res_code,
    current_sub_cat,
    current_fault_cat,
    short_desc,
    resolution_notes
):
    evidence_text = f"{short_desc} {resolution_notes}".lower()

    matched_rule = None

    for _, rule in rules.sort_values("priority").iterrows():
        rule_keywords = str(rule["keywords"]).lower().split("|")

        has_keyword = any(k in evidence_text for k in rule_keywords)
        has_exclusion = any(e in evidence_text for e in EXCLUSION_KEYWORDS)
        has_external = any(e in evidence_text for e in EXTERNAL_NETWORK_KEYWORDS)
        has_customer = any(c in evidence_text for c in CUSTOMER_PREMISES_KEYWORDS)

        # Special handling for Road Traffic Collision
        if rule["rule_id"] == "R23":
            if has_keyword and has_external and not has_customer:
                matched_rule = rule
                break
            else:
                continue

        if has_keyword and not has_exclusion:
            matched_rule = rule
            break

    if matched_rule is None:
        return {
            "decision": "AUTO-NO",
            "confidence": 0.0,
            "reason": "No SOP rule matched",
            "suggested": None
        }

    confidence = matched_rule["confidence"]

    needs_update = (
        str(current_res_code).strip() != str(matched_rule["output_resolution_code"]).strip()
        or str(current_sub_cat).strip() != str(matched_rule["output_sub_category"]).strip()
        or str(current_fault_cat).strip() != str(matched_rule["output_fault_category"]).strip()
    )

    if confidence < CONFIDENCE_THRESHOLD:
        decision = "REVIEW"
    elif needs_update:
        decision = "AUTO-YES"
    else:
        decision = "AUTO-NO"

    return {
        "decision": decision,
        "confidence": confidence,
        "reason": matched_rule["notes"],
        "suggested": {
            "Resolution Code": matched_rule["output_resolution_code"],
            "Sub Category": matched_rule["output_sub_category"],
            "Fault Category": matched_rule["output_fault_category"],
            "Rule ID": matched_rule["rule_id"]
        }
    }

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Resolution Code Decision Engine", layout="centered")

st.title("📋 Incident Resolution Decision Engine")

st.subheader("🔹 Input Incident Details")

inc_number = st.text_input("INC Number")

current_res_code = st.text_input("Current Resolution Code")
current_sub_cat = st.text_input("Current Resolution Sub Category")
current_fault_cat = st.text_input("Current Fault Category")

short_desc = st.text_area("Short Description", height=100)
resolution_notes = st.text_area("Resolution Notes", height=150)

if st.button("🔍 Evaluate Incident"):
    result = decide_incident(
        current_res_code,
        current_sub_cat,
        current_fault_cat,
        short_desc,
        resolution_notes
    )

    st.subheader("✅ Decision Result")

    st.markdown(f"### **Decision:** `{result['decision']}`")
    st.markdown(f"**Confidence:** `{round(result['confidence'], 2)}`")
    st.markdown(f"**Reason:** {result['reason']}")

    if result["suggested"]:
        st.subheader("🔁 Suggested Update")
        st.write(result["suggested"])

