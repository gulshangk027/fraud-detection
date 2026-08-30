"""
Official Bank Finalized Anchor Features Dictionary for MuleNet AI (Drishthi).
Contains exact official variable names and descriptions from Description.xlsx
as well as descriptive mappings for engineered behavioral features.
"""

BANK_FINALIZED_FEATURES = {
    "F115": {
        "variable": "R_CI_NON_CASH_CHQ_TXN_L14_31D",
        "description": "Ratio of Customer Induced Non Cash Non Cheque Total txns - last 14 to 31D",
        "category": "Ratio"
    },
    "F321": {
        "variable": "RA_NON_CASH_CHQ_AMT_L7_14D",
        "description": "Ratio of avgs: of Non Cash Non Cheque Total Amount - last 7 to 14D",
        "category": "Ratio"
    },
    "F527": {
        "variable": "RA_CI_NON_CASH_CHQ_TXN_CR_L7_31D",
        "description": "Ratio of avgs: of Customer Induced Non Cash Non Cheque Credit txns - last 7 to 31D",
        "category": "Ratio"
    },
    "F531": {
        "variable": "RA_CI_NON_CASH_CHQ_AMT_DB_L7_31D",
        "description": "Ratio of avgs: of Customer Induced Non Cash Non Cheque Debit Amount - last 7 to 31D",
        "category": "Ratio"
    },
    "F670": {
        "variable": "MIN_UPI_XFER_TXNS_L7D",
        "description": "Min UPI Total Txns - last 7D",
        "category": "UPI"
    },
    "F1692": {
        "variable": "CASH_TXNS_DB_L14D",
        "description": "Cash Debit Txns - last 14D",
        "category": "Cash"
    },
    "F2082": {
        "variable": "AVG_NET_BNKING_TXNS_DB_L14D",
        "description": "Average Net Banking Debit Txns - last 14D",
        "category": "Electronic Transfer"
    },
    "F2122": {
        "variable": "AVG_CASH_TXNS_L31D",
        "description": "Average Cash Transaction Count - last 31D",
        "category": "Cash"
    },
    "F2582": {
        "variable": "DA_UPI_TXN_CR_L7_14D",
        "description": "Deviation of avgs: of UPI Total Amount - last 7 to 14D",
        "category": "Deviation"
    },
    "F2678": {
        "variable": "DA_ELEC_XFER_AMT_L14_31D",
        "description": "Deviation of avgs: of Online Transfer (IMPS+NEFT+RTGS) Total Amount - last 14 to 31D",
        "category": "Deviation"
    },
    "F2737": {
        "variable": "DA_NON_CASH_CHQ_AMT_L7_31D",
        "description": "Deviation of avgs: of Non Cash Non Cheque Total Amount - last 7 to 31D",
        "category": "Deviation"
    },
    "F2956": {
        "variable": "D_TA_CI_NON_CASH_CHQ_TXN_CR_L14D",
        "description": "Deviation of total from avg: Customer Induced Non Cash Non Cheque Credit txns - last 14 to 31D",
        "category": "Deviation"
    },
    "F3043": {
        "variable": "D_TA_CASH_TXN_L31D",
        "description": "Deviation of total from avg: Cash Total txns - last 7 to 14D",
        "category": "Deviation"
    },
    "F3836": {
        "variable": "AVG_BAL_14DAYS",
        "description": "Average account balance in last 14 days",
        "category": "Account"
    },
    "F3887": {
        "variable": "TENURE_AS_OF_ALERT",
        "description": "Customer tenure with the bank as of the alert date",
        "category": "Customer"
    },
    "F3889": {
        "variable": "ACCT_OPN_DAYS",
        "description": "Number of days since account opened (buckets)",
        "category": "Account"
    },
    "F3891": {
        "variable": "CUST_OCCP",
        "description": "Occupation code of customer",
        "category": "Customer"
    },
    "F3894": {
        "variable": "AGE_IN_YRS",
        "description": "Customer age as of alert date",
        "category": "Customer"
    }
}

# Mapping for engineered behavioral signals
ENGINEERED_BEHAVIORAL_MAP = {
    "F1": {"variable": "Transaction Velocity", "description": "Rapid acceleration in transaction frequency (14D)", "category": "Behavioral"},
    "F2": {"variable": "Rapid Transfer Count", "description": "High-frequency outbound electronic transfers", "category": "Behavioral"},
    "F3": {"variable": "Unique Senders", "description": "Number of distinct incoming money senders (7D)", "category": "Behavioral"},
    "F4": {"variable": "Unique Receivers", "description": "Number of distinct outgoing money beneficiaries (7D)", "category": "Behavioral"},
    "F5": {"variable": "New Beneficiary Count", "description": "Newly added transfer beneficiaries within 24 hours", "category": "Behavioral"},
    "F6": {"variable": "Night Transaction Ratio", "description": "Percentage of transactions initiated between 12 AM - 5 AM", "category": "Behavioral"},
    "F7": {"variable": "Device Changes", "description": "Number of distinct mobile devices/IMEIs linked to account", "category": "Behavioral"},
    "F8": {"variable": "Geo Changes", "description": "Geographic IP/location variance during login/transactions", "category": "Behavioral"},
    "F9": {"variable": "Cash Withdrawal Activity", "description": "ATM & Branch cash withdrawal velocity after credit", "category": "Behavioral"},
    "F10": {"variable": "Credit-to-Debit Imbalance", "description": "Ratio of immediate debit drain vs total credit inflow", "category": "Behavioral"},
    "F11": {"variable": "IMPS Transfer Velocity", "description": "Outbound IMPS high-speed transfer frequency", "category": "Behavioral"},
    "F12": {"variable": "Branch Cash Inflow Ratio", "description": "Ratio of branch cash deposits vs digital transfers", "category": "Behavioral"},
    "F13": {"variable": "ATM Cash Withdrawal Velocity", "description": "ATM cash withdrawal velocity following credit burst", "category": "Behavioral"},
    "F14": {"variable": "Beneficiary Addition Count", "description": "Frequency of newly added transfer payees", "category": "Behavioral"},
    "F15": {"variable": "Micro-Deposit Inflow Ratio", "description": "Proportion of incoming micro-amount test transfers", "category": "Behavioral"},
    "F16": {"variable": "Location Variance Index", "description": "Discrepancy in geographical login coordinates", "category": "Behavioral"},
    "F17": {"variable": "Mobile Session Velocity", "description": "Rapid mobile app session initiations", "category": "Behavioral"},
    "F18": {"variable": "API Gateway Transfer Frequency", "description": "Automated API transfer requests", "category": "Behavioral"},
    "F19": {"variable": "Cheque Clearance Velocity", "description": "High-value cheque deposit and rapid clearing", "category": "Behavioral"},
    "F20": {"variable": "High-Value Debit Drain Ratio", "description": "Percentage of account balance drained within 1 hour", "category": "Behavioral"}
}

TARGET_VARIABLE = "F3924"
TARGET_NAME = "FRAUD_TGT"

def get_feature_info(fname: str) -> dict:
    """Returns metadata dictionary for feature ID."""
    if fname in BANK_FINALIZED_FEATURES:
        info = BANK_FINALIZED_FEATURES[fname].copy()
        info["is_bank_finalized"] = True
        return info
    elif fname in ENGINEERED_BEHAVIORAL_MAP:
        info = ENGINEERED_BEHAVIORAL_MAP[fname].copy()
        info["is_bank_finalized"] = False
        return info
    else:
        # Format any unmapped feature ID into a clean human-readable string
        clean_name = fname
        if fname.startswith("F") and fname[1:].isdigit():
            clean_name = f"Engineered Behavioral Signal {fname}"
        return {
            "variable": clean_name,
            "description": f"Engineered Behavioral Signal {fname}",
            "category": "Behavioral Signal",
            "is_bank_finalized": False
        }
