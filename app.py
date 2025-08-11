import streamlit as st
import pandas as pd
import plotly as pt
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import io

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

POLICY_CSV = DATA_DIR / "policies.csv"
CLAIMS_CSV = DATA_DIR / "claims.csv"

st.set_page_config(page_title="Nath Investment Dashboard", layout="wide")

# ---------- sample data generator ----------
def generate_sample_data():
    if POLICY_CSV.exists() and CLAIMS_CSV.exists():
        return

    np.random.seed(42)
    n_policies = 250
    start_date = datetime.today() - timedelta(days=365)
    policy_types = ["Term Life", "ULIP", "Health", "Motor", "Home"]
    statuses = ["Active", "Lapsed", "Renewed"]

    policies = []
    for i in range(1, n_policies + 1):
        issued = start_date + timedelta(days=int(np.random.rand() * 365))
        tenure_years = int(np.random.choice([1,2,5,10,15,20], p=[0.1,0.1,0.35,0.2,0.15,0.1]))
        expiry = issued + timedelta(days=365*tenure_years)
        premium = float(round(np.random.uniform(3000, 60000), 2))
        ptype = np.random.choice(policy_types, p=[0.25,0.15,0.25,0.2,0.15])
        status = np.random.choice(statuses, p=[0.75,0.15,0.1])
        customer_age = int(np.random.uniform(20, 65))
        policies.append({
            "policy_id": f"P{i:04d}",
            "customer_name": f"Client {i}",
            "phone": f"9{np.random.randint(10**8,10**9-1)}",
            "policy_type": ptype,
            "issued_date": issued.date().isoformat(),
            "expiry_date": expiry.date().isoformat(),
            "tenure_years": tenure_years,
            "annual_premium": premium,
            "status": status,
            "customer_age": customer_age,
            "agent": np.random.choice(["You", "Agent A", "Agent B"], p=[0.6,0.25,0.15])
        })

    policies_df = pd.DataFrame(policies)
    policies_df.to_csv(POLICY_CSV, index=False)

    # claims: a subset of policies will have claims
    claims = []
    claim_reasons = ["Hospitalization", "Accident", "Theft", "Fire", "Other"]
    for _, row in policies_df.sample(frac=0.12).iterrows():
        claim_date = datetime.fromisoformat(row["issued_date"]) + timedelta(days=int(np.random.rand()*365))
        claims.append({
            "claim_id": f"C{np.random.randint(10000,99999)}",
            "policy_id": row["policy_id"],
            "claim_date": claim_date.date().isoformat(),
            "amount": round(row["annual_premium"] * np.random.uniform(0.3, 4.0), 2),
            "reason": np.random.choice(claim_reasons),
            "status": np.random.choice(["Settled","Pending","Rejected"], p=[0.6,0.3,0.1])
        })
    claims_df = pd.DataFrame(claims)
    claims_df.to_csv(CLAIMS_CSV, index=False)

generate_sample_data()

# ---------- load data ----------
@st.cache_data
def load_data():
    policies = pd.read_csv(POLICY_CSV, parse_dates=["issued_date","expiry_date"])
    claims = pd.read_csv(CLAIMS_CSV, parse_dates=["claim_date"]) if CLAIMS_CSV.exists() else pd.DataFrame()
    return policies, claims

policies, claims = load_data()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
agent_filter = st.sidebar.multiselect("Agent", options=sorted(policies["agent"].unique()), default=sorted(policies["agent"].unique()))
ptype_filter = st.sidebar.multiselect("Policy Type", options=sorted(policies["policy_type"].unique()), default=sorted(policies["policy_type"].unique()))
status_filter = st.sidebar.multiselect("Policy Status", options=sorted(policies["status"].unique()), default=sorted(policies["status"].unique()))
date_from = st.sidebar.date_input("Issued After", value=(datetime.today() - timedelta(days=365)).date())
date_to = st.sidebar.date_input("Issued Before", value=datetime.today().date())

mask = (
    policies["agent"].isin(agent_filter) &
    policies["policy_type"].isin(ptype_filter) &
    policies["status"].isin(status_filter) &
    (policies["issued_date"].dt.date >= date_from) &
    (policies["issued_date"].dt.date <= date_to)
)
filtered = policies[mask].copy()

# ---------- KPIs ----------
st.title("🏢 Nath Investment Dashboard")
col1, col2, col3, col4 = st.columns(4)
total_policies = len(filtered)
total_premium = filtered["annual_premium"].sum()
avg_premium = filtered["annual_premium"].mean() if total_policies else 0
active_policies = len(filtered[filtered["status"]=="Active"])
col1.metric("Policies", total_policies)
col2.metric("Total Annual Premium (₹)", f"{total_premium:,.2f}")
col3.metric("Avg Premium (₹)", f"{avg_premium:,.2f}")
col4.metric("Active Policies", active_policies)

# ---------- Charts ----------
st.markdown("### 📈 Premium by Policy Type")
premium_by_type = filtered.groupby("policy_type")["annual_premium"].sum().reset_index().sort_values("annual_premium", ascending=False)
fig1 = px.bar(premium_by_type, x="policy_type", y="annual_premium", labels={"annual_premium":"Total Annual Premium (₹)","policy_type":"Policy Type"})
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 🧾 Policies Issued Over Time")
issued = filtered.copy()
issued["month"] = issued["issued_date"].dt.to_period("M").dt.to_timestamp()
issued_ts = issued.groupby("month").size().reset_index(name="count")
fig2 = px.line(issued_ts, x="month", y="count", markers=True)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 🧾 Policy Count by Age Group")
bins = [0,25,35,45,55,100]
labels = ["<25","25-34","35-44","45-54","55+"]
filtered["age_group"] = pd.cut(filtered["customer_age"], bins=bins, labels=labels, right=False)
age_counts = filtered.groupby("age_group").size().reset_index(name="count")
fig3 = px.pie(age_counts, names="age_group", values="count", title="Customer Age Groups")
st.plotly_chart(fig3, use_container_width=True)

# ---------- Claims summary ----------
st.markdown("### 🚑 Claims Overview")
if not claims.empty:
    # join claims to filtered policies
    claims_f = claims.merge(filtered[["policy_id"]], on="policy_id")
    claims_by_status = claims_f.groupby("status")["amount"].sum().reset_index()
    fig4 = px.bar(claims_by_status, x="status", y="amount", labels={"amount":"Claim Amount (₹)"})
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(claims_f.sort_values("claim_date", ascending=False).reset_index(drop=True))
else:
    st.info("No claims data available.")

# ---------- Table of policies ----------
st.markdown("### 📋 Filtered Policies")
show_df = filtered[[
    "policy_id","customer_name","phone","policy_type","issued_date","expiry_date","tenure_years","annual_premium","status","agent"
]].sort_values("issued_date", ascending=False).reset_index(drop=True)
st.dataframe(show_df)

# ---------- CSV / Excel export ----------
st.markdown("### 📥 Export Data")
def to_excel_bytes(df: pd.DataFrame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="policies")
    return buffer.getvalue()

if st.button("Export Filtered Policies to Excel"):
    b = to_excel_bytes(show_df)
    st.download_button("Download Excel", data=b, file_name=f"policies_{datetime.now().date()}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- Quick actions ----------
st.markdown("### ⚡ Quick Actions")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Show Top 10 Premium Policies"):
        top10 = policies.sort_values("annual_premium", ascending=False).head(10)
        st.table(top10[["policy_id","customer_name","policy_type","annual_premium"]])
with col_b:
    if st.button("Show Renewals Next 30 days"):
        today = datetime.today().date()
        soon = policies[(pd.to_datetime(policies["expiry_date"]).dt.date >= today) &
                        (pd.to_datetime(policies["expiry_date"]).dt.date <= today + timedelta(days=30))]
        st.table(soon[["policy_id","customer_name","expiry_date","phone"]])

st.markdown("---")
st.caption("Template: customize fields, add agent login, WhatsApp reminders, or import your CSVs into data/ folder.")
