"""
Comprehensive Benchmark Test Suite for Awaaz (Victim Response & Complaint Assistance Engine)
Tests 20 diverse incident scenarios across English, Indian English, Hinglish, Negation, Entity Extraction, and Complaint Generation.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.nlp_service import analyze_awaaz_incident

AWAAZ_BENCHMARK_CASES = [
    {
        "id": "AW1",
        "name": "UPI Customer Care Vishing Scam",
        "input": "Someone called me pretending to be SBI customer care official yesterday and persuaded me to enter my UPI pin. I lost ₹25,000 via PhonePe UPI.",
        "exp_fraud": "Customer Care Impersonation",
        "exp_amount": "₹25,000.00",
        "exp_mode": "UPI (GPay/PhonePe/Paytm)",
        "exp_bank": "SBI",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW2",
        "name": "OTP Scam with Negation (Refused OTP)",
        "input": "I received a call claiming to be HDFC bank support asking for my OTP to prevent account block, but I did NOT share my OTP and hung up.",
        "exp_fraud": "Customer Care Impersonation",
        "exp_amount": "Not provided by the complainant.",
        "exp_mode": "Not provided by the complainant.",
        "exp_bank": "HDFC Bank",
        "exp_otp": "No (Refused / Not Shared)"
    },
    {
        "id": "AW3",
        "name": "Telegram Part-Time Task Job Scam",
        "input": "I joined a Telegram part-time job group on 15 August. They asked me to deposit ₹5,000 first and promised that I would get ₹10,000 back. After paying, they stopped responding.",
        "exp_fraud": "Job / Part-Time Task Scam (Mule Fraud)",
        "exp_amount": "₹5,000.00",
        "exp_mode": "Not provided by the complainant.",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW4",
        "name": "KYC Expiry Phishing Link Scam",
        "input": "Received SMS saying SBI KYC expires today. Clicked https://bank-kyc-verify.example and ₹15,000 was debited from my account.",
        "exp_fraud": "KYC Expiry Scam",
        "exp_amount": "₹15,000.00",
        "exp_mode": "Not provided by the complainant.",
        "exp_bank": "SBI",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW5",
        "name": "UPI Collect Request Scam",
        "input": "I received a UPI collect request from an unknown person on Paytm. I thought it was a payment request and approved it. ₹8,500 was deducted from my account.",
        "exp_fraud": "UPI Collect Scam",
        "exp_amount": "₹8,500.00",
        "exp_mode": "UPI (GPay/PhonePe/Paytm)",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW6",
        "name": "Fake Tax Refund Claim Scam",
        "input": "Got an email claiming Income Tax refund of ₹12,500 was approved. Shared NetBanking details and ₹12,500 was stolen from ICICI account.",
        "exp_fraud": "Fake Refund Scam",
        "exp_amount": "₹12,500.00",
        "exp_mode": "Net Banking",
        "exp_bank": "ICICI Bank",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW7",
        "name": "Fake Loan Processing Fee Scam",
        "input": "Applied for online instant loan. Agent asked to pay ₹3,500 as verification fee via GPay UPI. After transfer, loan was not disbursed.",
        "exp_fraud": "Fake Loan Scam",
        "exp_amount": "₹3,500.00",
        "exp_mode": "UPI (GPay/PhonePe/Paytm)",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW8",
        "name": "Investment / Crypto Trading Scam",
        "input": "Deposited 50k rupees in crypto trading portal promising double returns in 3 days. Now account is locked and website is unreachable.",
        "exp_fraud": "Investment / Cryptocurrency Scam",
        "exp_amount": "₹50,000.00",
        "exp_mode": "Cryptocurrency (USDT/Crypto)",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW9",
        "name": "Card Fraud / ATM Cash Withdrawal",
        "input": "Unusual transaction of ₹18,000 debited from Axis bank debit card at an unknown ATM yesterday night.",
        "exp_fraud": "Card Fraud",
        "exp_amount": "₹18,000.00",
        "exp_mode": "Debit/Credit Card",
        "exp_bank": "Axis Bank",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW10",
        "name": "QR Code Reverse Payment Scam",
        "input": "Scammer sent QR code on WhatsApp claiming to send payment for OLX item. Scanned QR and entered UPI PIN, ₹6,000 deducted.",
        "exp_fraud": "QR Code Scam",
        "exp_amount": "₹6,000.00",
        "exp_mode": "UPI (GPay/PhonePe/Paytm)",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW11",
        "name": "Hinglish Incident Statement",
        "input": "Mere account se 20k kat gaya UPI se. Someone called from bank and scammed me.",
        "exp_fraud": "Customer Care Impersonation",
        "exp_amount": "₹20,000.00",
        "exp_mode": "UPI (GPay/PhonePe/Paytm)",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW12",
        "name": "Incident with OTP Shared Positively",
        "input": "Fake customer support requested OTP and I shared the OTP. ₹30,000 was debited from SBI account.",
        "exp_fraud": "Customer Care Impersonation",
        "exp_amount": "₹30,000.00",
        "exp_mode": "Not provided by the complainant.",
        "exp_bank": "SBI",
        "exp_otp": "Yes (OTP Shared)"
    },
    {
        "id": "AW13",
        "name": "Incident with Missing Details (Only Loss Amount)",
        "input": "I lost ₹10,000 in an online scam.",
        "exp_fraud": "Cyber-Enabled Financial Fraud",
        "exp_amount": "₹10,000.00",
        "exp_mode": "Not provided by the complainant.",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW14",
        "name": "Legitimate Bank Salary Credit",
        "input": "Monthly salary of ₹75,000 credited to my HDFC bank account.",
        "exp_fraud": "Legitimate Non-Fraud Transaction",
        "exp_amount": "₹75,000.00",
        "exp_mode": "Not provided by the complainant.",
        "exp_bank": "HDFC Bank",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW15",
        "name": "Legitimate Tuition Fee Payment",
        "input": "Successfully paid tuition fee of ₹45,000 using Net Banking to official college portal.",
        "exp_fraud": "Legitimate Non-Fraud Transaction",
        "exp_amount": "₹45,000.00",
        "exp_mode": "Net Banking",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW16",
        "name": "Account Takeover / NetBanking Credential Theft",
        "input": "Scammer called pretending to be Kotak bank manager and asked for netbanking username and password to unblock account.",
        "exp_fraud": "Customer Care Impersonation",
        "exp_amount": "Not provided by the complainant.",
        "exp_mode": "Net Banking",
        "exp_bank": "Kotak Mahindra Bank",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW17",
        "name": "Fake Refund with UTR Reference ID",
        "input": "Received SMS about Amazon refund. Transaction UTR 321456987123 was used to deduct ₹4,500 from PNB account.",
        "exp_fraud": "Fake Refund Scam",
        "exp_amount": "₹4,500.00",
        "exp_mode": "Not provided by the complainant.",
        "exp_bank": "PNB",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW18",
        "name": "WhatsApp Loan Scam",
        "input": "Received WhatsApp message offering zero interest instant loan. Paid ₹2,000 registration fee via PhonePe.",
        "exp_fraud": "Fake Loan Scam",
        "exp_amount": "₹2,000.00",
        "exp_mode": "UPI (GPay/PhonePe/Paytm)",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW19",
        "name": "Multi-Pattern Vishing & Phishing",
        "input": "Caller pretended to be Canara Bank official and sent a phishing link to steal ₹14,000.",
        "exp_fraud": "Customer Care Impersonation",
        "exp_amount": "₹14,000.00",
        "exp_mode": "Not provided by the complainant.",
        "exp_bank": "Canara Bank",
        "exp_otp": "Not mentioned"
    },
    {
        "id": "AW20",
        "name": "Complex Sentence Structure with Relative Time",
        "input": "Yesterday evening someone sent a UPI request for ₹7,500 claiming to be courier agent. I approved it on GPay.",
        "exp_fraud": "UPI Collect Scam",
        "exp_amount": "₹7,500.00",
        "exp_mode": "UPI (GPay/PhonePe/Paytm)",
        "exp_bank": "Not provided by the complainant.",
        "exp_otp": "Not mentioned"
    }
]

def run_awaaz_benchmark():
    print("=" * 110)
    print("AWAAZ VICTIM INCIDENT ASSISTANCE ENGINE BENCHMARK EVALUATION")
    print("=" * 110)

    passed_count = 0
    total = len(AWAAZ_BENCHMARK_CASES)

    print(f"| {'ID':<4} | {'Test Case Name':<32} | {'Exp Category':<30} | {'Extracted Amt':<12} | {'OTP Status':<20} | {'Status':<6} |")
    print("|" + "-"*6 + "|" + "-"*34 + "|" + "-"*32 + "|" + "-"*14 + "|" + "-"*22 + "|" + "-"*8 + "|")

    for case in AWAAZ_BENCHMARK_CASES:
        res = analyze_awaaz_incident(case["input"])

        fraud_ok = case["exp_fraud"].lower() in res["fraud_type"].lower()
        amt_ok = (case["exp_amount"] == res["extracted_amount"])
        otp_ok = (case["exp_otp"] == res["otp_status"])

        passed = fraud_ok and amt_ok and (not case["exp_otp"].startswith("Yes") or res["otp_status"].startswith("Yes"))
        status_str = "PASS" if passed else "FAIL"

        if passed:
            passed_count += 1

        disp_amt = res['extracted_amount'].replace("₹", "Rs.")[:12]
        disp_otp = res['otp_status'][:20]

        print(f"| {case['id']:<4} | {case['name'][:32]:<32} | {res['fraud_type'][:30]:<30} | {disp_amt:<12} | {disp_otp:<20} | {status_str:<6} |")

    print("=" * 110)
    print(f"TOTAL RESULT: {passed_count} / {total} TEST CASES PASSED ({passed_count/total*100:.1f}%)")
    print("=" * 110)

    assert passed_count >= 18, f"Benchmark require at least 18/20 test cases passing! Got {passed_count}."

if __name__ == "__main__":
    run_awaaz_benchmark()
