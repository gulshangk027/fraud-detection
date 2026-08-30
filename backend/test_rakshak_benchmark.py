"""
Rakshak Benchmark Test Suite (TC1 - TC80)
Evaluates Rakshak NLP engine accuracy, score tolerance, and risk level assignment.
"""

import sys
import os

# Ensure backend app is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.nlp_service import analyze_rakshak_message

BENCHMARK_CASES = [
    # TC1 - TC50: Existing Benchmark
    {"tc": "TC1", "name": "Work from home job", "message": "Congratulations! You have been selected for a work-from-home job. Earn ₹2,000-₹5,000 daily.", "expected_level": "HIGH", "target_score": 82, "tol": 7},
    {"tc": "TC2", "name": "Business payments account", "message": "Need a bank account for receiving business payments. Earn 5% commission daily.", "expected_level": "CRITICAL", "target_score": 95, "tol": 5},
    {"tc": "TC3", "name": "Commission transfer mule", "message": "Transfer RS 2,00,000 to your account. Keep 10% commission and transfer RS 1,80,000 to manager.", "expected_level": "CRITICAL", "target_score": 97, "tol": 5},
    {"tc": "TC4", "name": "Urgent account transfer", "message": "URGENT: Please receive RS 75,000 in your bank account today.", "expected_level": "CRITICAL", "target_score": 96, "tol": 5},
    {"tc": "TC5", "name": "SBI KYC Expired", "message": "SBI bank account KYC has expired. Contact support to update.", "expected_level": "CRITICAL", "target_score": 98, "tol": 5},
    {"tc": "TC6", "name": "Crypto USDT Binance", "message": "Buy USDT crypto on Binance and transfer to our wallet.", "expected_level": "CRITICAL", "target_score": 94, "tol": 5},
    {"tc": "TC7", "name": "Unrealistic Trading Returns", "message": "Guaranteed 500% returns in 24 hours. Deposit in our trading account.", "expected_level": "CRITICAL", "target_score": 99, "tol": 5},
    {"tc": "TC8", "name": "Student Part-Time Job", "message": "Student part-time job available: Receive customer payments in your personal account.", "expected_level": "HIGH", "target_score": 86, "tol": 7},
    {"tc": "TC9", "name": "Amazon Refund Scam", "message": "Amazon refund of RS 4,999 pending. Send bank details and OTP.", "expected_level": "HIGH", "target_score": 91, "tol": 5},
    {"tc": "TC10", "name": "HDFC Bank Support Impersonation", "message": "This is HDFC bank customer support. Unusual transaction detected on your debit card.", "expected_level": "CRITICAL", "target_score": 97, "tol": 5},
    {"tc": "TC11", "name": "Layered UPI Split", "message": "Split into 3 parts of RS 50,000 and transfer to 3 different UPI IDs.", "expected_level": "CRITICAL", "target_score": 99, "tol": 5},
    {"tc": "TC12", "name": "YouTube Like Task Scam", "message": "Earn money by liking YouTube videos. Registration fee ₹500 required.", "expected_level": "HIGH", "target_score": 84, "tol": 7},
    {"tc": "TC13", "name": "KBC Lucky Draw Fee", "message": "Kaun Banega Crorepati lucky draw winner! Pay processing fee to claim.", "expected_level": "HIGH", "target_score": 89, "tol": 7},
    {"tc": "TC14", "name": "International Remittance Cut", "message": "Receive $2,000 international remittance. Deduct fee and transfer balance.", "expected_level": "HIGH", "target_score": 88, "tol": 7},
    {"tc": "TC15", "name": "UPI Collect Request", "message": "Approve the RS 20,000 UPI collect request.", "expected_level": "CRITICAL", "target_score": 96, "tol": 5},
    {"tc": "TC16", "name": "Credential Harvesting", "message": "Hand over your bank account login credentials for business transactions.", "expected_level": "CRITICAL", "target_score": 100, "tol": 5},
    {"tc": "TC17", "name": "Pre-Approved Loan Advance Fee", "message": "Instant pre-approved loan of RS 5,00,000 sanctioned. Transfer processing charge.", "expected_level": "CRITICAL", "target_score": 99, "tol": 5},
    {"tc": "TC18", "name": "Remote Writer Job", "message": "Remote content writer post: Send your resume and portfolio to careers@techcorp.com.", "expected_level": "MODERATE", "target_score": 32, "tol": 5},
    {"tc": "TC19", "name": "Branch Cash Deposit", "message": "Deposit RS 50,000 cash at local bank branch and receive instant UPI transfer.", "expected_level": "CRITICAL", "target_score": 98, "tol": 5},
    {"tc": "TC20", "name": "Urgent Bank Account Password", "message": "URGENT JOB: Open new account in bank and hand over password.", "expected_level": "CRITICAL", "target_score": 98, "tol": 5},
    {"tc": "TC21", "name": "Work From Home Processor", "message": "Work from home payment processor needed to collect company sales in personal account.", "expected_level": "CRITICAL", "target_score": 93, "tol": 5},
    {"tc": "TC22", "name": "ICICI Account Suspended", "message": "ICICI Bank notice: Your account is suspended due to pending re-KYC.", "expected_level": "CRITICAL", "target_score": 99, "tol": 5},
    {"tc": "TC23", "name": "Vendor Fund Transfer", "message": "Deposit RS 3,00,000 in your account and transfer RS 2,70,000 to vendor.", "expected_level": "CRITICAL", "target_score": 97, "tol": 5},
    {"tc": "TC24", "name": "Unknown Sender Transfer", "message": "Unknown sender wants to transfer funds to your account.", "expected_level": "HIGH", "target_score": 78, "tol": 7},
    {"tc": "TC25", "name": "Flipkart Refund", "message": "Flipkart refund of RS 2,499 approved. Share bank details.", "expected_level": "HIGH", "target_score": 88, "tol": 7},
    {"tc": "TC26", "name": "Company Sales Collection", "message": "Personal savings accounts to collect company sales from clients across India.", "expected_level": "CRITICAL", "target_score": 94, "tol": 5},
    {"tc": "TC27", "name": "Freelance Overpayment Refund", "message": "Freelance project payment: Client will send RS 80,000 refund overpayment back.", "expected_level": "HIGH", "target_score": 83, "tol": 7},
    {"tc": "TC28", "name": "Data Entry Registration Fee", "message": "Selected for data entry operator role. Pay ₹1,500 security deposit for laptop.", "expected_level": "HIGH", "target_score": 81, "tol": 7},
    {"tc": "TC29", "name": "Income Tax Refund", "message": "Income tax refund of RS 18,450 approved.", "expected_level": "HIGH", "target_score": 90, "tol": 5},
    {"tc": "TC30", "name": "WazirX Crypto Trader", "message": "Crypto trader looking for account holders to buy Bitcoin on WazirX.", "expected_level": "CRITICAL", "target_score": 96, "tol": 5},
    {"tc": "TC31", "name": "Multiple Bank Accounts", "message": "Need 3 bank accounts to distribute company client payments.", "expected_level": "CRITICAL", "target_score": 98, "tol": 5},
    {"tc": "TC32", "name": "ATM Cash Withdrawal", "message": "Withdraw cash from local ATM and hand over cash to field agent.", "expected_level": "CRITICAL", "target_score": 95, "tol": 5},
    {"tc": "TC33", "name": "Rent UPI ID Gaming", "message": "Rent your UPI ID & PhonePe account to receive online gaming app funds.", "expected_level": "CRITICAL", "target_score": 100, "tol": 5},
    {"tc": "TC34", "name": "Pool Funds Senders", "message": "Receive money from 10 different senders, pool funds and forward.", "expected_level": "CRITICAL", "target_score": 99, "tol": 5},
    {"tc": "TC35", "name": "LIC Policy Bonus", "message": "LIC policy bonus of RS 1,20,000 released.", "expected_level": "HIGH", "target_score": 87, "tol": 7},
    {"tc": "TC36", "name": "Disaster Relief Funds", "message": "Help collect emergency disaster relief funds in personal account.", "expected_level": "HIGH", "target_score": 76, "tol": 7},
    {"tc": "TC37", "name": "University Scholarship Fee", "message": "Guaranteed international university scholarship of RS 2,00,000.", "expected_level": "HIGH", "target_score": 85, "tol": 7},
    {"tc": "TC38", "name": "Ludo App Payment Agent", "message": "Online ludo app needs payment agents to receive player deposits.", "expected_level": "HIGH", "target_score": 82, "tol": 7},
    {"tc": "TC39", "name": "Foreign Wire Forwarding", "message": "Receive foreign inward wire transfer of $1,500.", "expected_level": "HIGH", "target_score": 86, "tol": 7},
    {"tc": "TC40", "name": "Courier Delivery Charge", "message": "Courier delivery failed. Pay fee for re-delivery.", "expected_level": "HIGH", "target_score": 89, "tol": 7},
    {"tc": "TC41", "name": "Tuition Fee Paid Safeguard", "message": "Your tuition fee of ₹45,000 was successfully paid to college.", "expected_level": "LOW", "target_score": 4, "tol": 5},
    {"tc": "TC42", "name": "Electricity Bill Safeguard", "message": "Electricity bill payment credited to your bank account.", "expected_level": "LOW", "target_score": 2, "tol": 5},
    {"tc": "TC43", "name": "Salary Credit Safeguard", "message": "Monthly salary credited to your registered bank account.", "expected_level": "LOW", "target_score": 5, "tol": 5},
    {"tc": "TC44", "name": "Invoice Safeguard", "message": "Invoice #INV-2026-884 for consulting services.", "expected_level": "LOW", "target_score": 8, "tol": 5},
    {"tc": "TC45", "name": "Order Confirmed Safeguard", "message": "Order confirmed: Your purchase on e-commerce is processed.", "expected_level": "LOW", "target_score": 3, "tol": 5},
    {"tc": "TC46", "name": "Negation Safety Warning", "message": "Bank Security Warning: Never share your OTP, PIN, or banking password.", "expected_level": "LOW", "target_score": 10, "tol": 5},
    {"tc": "TC47", "name": "Debit Card Spent Safeguard", "message": "Spent on your debit card at store.", "expected_level": "LOW", "target_score": 2, "tol": 5},
    {"tc": "TC48", "name": "Official Disbursement Safeguard", "message": "Disbursement status at official portal updated.", "expected_level": "LOW", "target_score": 4, "tol": 5},
    {"tc": "TC49", "name": "Borderline Remote Writer", "message": "Remote content writer post: Submit resume and past articles for evaluation.", "expected_level": "MODERATE", "target_score": 32, "tol": 5},
    {"tc": "TC50", "name": "Borderline Invoicing Assistant", "message": "Assistant needed to invoice international clients and convert USD to INR.", "expected_level": "MODERATE", "target_score": 35, "tol": 5},

    # TC51 - TC80: Link-Based Benchmark
    {"tc": "TC51", "name": "Fake Bank KYC Link", "message": "https://bank-kyc-verify.example/secure — Your KYC expires today. Verify immediately.", "expected_level": "CRITICAL", "target_score": 99, "tol": 5},
    {"tc": "TC52", "name": "Fake UPI Refund Link", "message": "https://upi-refund.example/claim — Click to receive your ₹8,500 refund.", "expected_level": "CRITICAL", "target_score": 97, "tol": 5},
    {"tc": "TC53", "name": "Fake SBI Login", "message": "https://sbi-login-security.example/ — Your account will be blocked. Login now.", "expected_level": "CRITICAL", "target_score": 100, "tol": 5},
    {"tc": "TC54", "name": "Fake Tax Refund", "message": "https://income-tax-refund.example/verify — Claim your pending refund.", "expected_level": "HIGH", "target_score": 91, "tol": 5},
    {"tc": "TC55", "name": "Fake Loan Approval", "message": "https://loan-approved.example/disbursement — Pay verification fee to release your loan.", "expected_level": "HIGH", "target_score": 94, "tol": 5},
    {"tc": "TC56", "name": "Fake Scholarship Link", "message": "https://student-scholarship.example/apply — Enter bank details to receive scholarship.", "expected_level": "HIGH", "target_score": 90, "tol": 5},
    {"tc": "TC57", "name": "Fake Job Portal", "message": "https://work-from-home-payment.example/register — Register your bank account to start earning.", "expected_level": "HIGH", "target_score": 88, "tol": 7},
    {"tc": "TC58", "name": "Payment Verification Link", "message": "https://payment-check.example/verify — Verify your account to receive payment.", "expected_level": "CRITICAL", "target_score": 96, "tol": 5},
    {"tc": "TC59", "name": "Fake Delivery Refund", "message": "https://delivery-refund.example/refund — Confirm your bank details for refund.", "expected_level": "HIGH", "target_score": 89, "tol": 7},
    {"tc": "TC60", "name": "Suspicious Shortened Link", "message": "https://short.example/a8K2p — Important! Open this link to verify your account.", "expected_level": "HIGH", "target_score": 84, "tol": 7},
    {"tc": "TC61", "name": "Fake Customer Support", "message": "https://support-account.example/help — Contact support and verify your banking information.", "expected_level": "CRITICAL", "target_score": 95, "tol": 5},
    {"tc": "TC62", "name": "Fake Investment Dashboard", "message": "https://investment-profit.example/withdraw — Withdraw your ₹75,000 profit.", "expected_level": "CRITICAL", "target_score": 98, "tol": 5},
    {"tc": "TC63", "name": "Fake Crypto Wallet", "message": "https://crypto-wallet-verify.example/connect — Connect wallet to receive your funds.", "expected_level": "HIGH", "target_score": 87, "tol": 7},
    {"tc": "TC64", "name": "Fake Bank Reward", "message": "https://bank-reward.example/claim — You have received a special cashback reward.", "expected_level": "HIGH", "target_score": 86, "tol": 7},
    {"tc": "TC65", "name": "Fake PAN Verification", "message": "https://pan-update.example/verify — Update PAN details to avoid account restrictions.", "expected_level": "CRITICAL", "target_score": 96, "tol": 5},
    {"tc": "TC66", "name": "Fake Aadhaar Verification", "message": "https://aadhaar-check.example/update — Complete verification to continue banking services.", "expected_level": "CRITICAL", "target_score": 97, "tol": 5},
    {"tc": "TC67", "name": "Fake UPI Collect Request", "message": "https://upi-collect.example/request — Approve this request to receive money.", "expected_level": "CRITICAL", "target_score": 95, "tol": 5},
    {"tc": "TC68", "name": "Fake E-commerce Offer", "message": "https://mega-sale.example/order — ₹1,999 product available for ₹99. Pay now.", "expected_level": "HIGH", "target_score": 79, "tol": 7},
    {"tc": "TC69", "name": "Fake Prize Link", "message": "https://winner-prize.example/claim — Congratulations! Claim ₹5 lakh.", "expected_level": "HIGH", "target_score": 93, "tol": 5},
    {"tc": "TC70", "name": "Fake Insurance Claim", "message": "https://insurance-refund.example/claim — Complete bank verification to receive your claim.", "expected_level": "HIGH", "target_score": 90, "tol": 5},
    {"tc": "TC71", "name": "Suspicious Payment Gateway", "message": "https://secure-payment.example/pay?id=83921 — Complete payment within 10 minutes.", "expected_level": "HIGH", "target_score": 82, "tol": 7},
    {"tc": "TC72", "name": "Lookalike Bank Domain", "message": "https://bankname-secure.example/login — Unusual activity detected. Login immediately.", "expected_level": "CRITICAL", "target_score": 99, "tol": 5},
    {"tc": "TC73", "name": "Login + OTP Request", "message": "https://account-security.example/login — Enter username, password and OTP.", "expected_level": "CRITICAL", "target_score": 100, "tol": 5},
    {"tc": "TC74", "name": "Account Unlock Link", "message": "https://account-unlock.example/verify — Your account is temporarily locked. Verify now.", "expected_level": "CRITICAL", "target_score": 98, "tol": 5},
    {"tc": "TC75", "name": "Legitimate College Portal", "message": "https://college.example/student-fees — View your fee payment status.", "expected_level": "LOW", "target_score": 3, "tol": 5},
    {"tc": "TC76", "name": "Legitimate Government Portal", "message": "https://government.example/scholarship/status — Check your application status.", "expected_level": "LOW", "target_score": 4, "tol": 5},
    {"tc": "TC77", "name": "Legitimate Bank Website", "message": "https://bank.example/ — Visit the official website to view your account.", "expected_level": "LOW", "target_score": 2, "tol": 5},
    {"tc": "TC78", "name": "Legitimate E-commerce Order", "message": "https://shop.example/orders — Track your recent order.", "expected_level": "LOW", "target_score": 3, "tol": 5},
    {"tc": "TC79", "name": "Borderline Job Link", "message": "https://jobs.example/payment-assistant — View job description and application details.", "expected_level": "MODERATE", "target_score": 25, "tol": 5},
    {"tc": "TC80", "name": "Borderline Payment Portal", "message": "https://business.example/invoice/83921 — View the invoice and payment status.", "expected_level": "LOW", "target_score": 12, "tol": 5}
]

def run_benchmark():
    print("=" * 110)
    print("RAKSHAK BENCHMARK EVALUATION (TC1 - TC80)")
    print("=" * 110)
    
    passed_count = 0
    total_count = len(BENCHMARK_CASES)
    outside_tolerance = []
    
    print(f"| {'TC':<6} | {'Test Case Name':<32} | {'Target':<6} | {'Actual':<6} | {'Diff':<5} | {'Exp Level':<10} | {'Act Level':<10} | {'Status':<6} |")
    print("|" + "-"*8 + "|" + "-"*34 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*7 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*8 + "|")
    
    for case in BENCHMARK_CASES:
        res = analyze_rakshak_message(case["message"])
        actual_score = res["risk_score"]
        actual_level = res["risk_level"]
        target_score = case["target_score"]
        expected_level = case["expected_level"]
        tol = case["tol"]
        
        diff = actual_score - target_score
        score_ok = abs(diff) <= tol
        level_ok = (actual_level == expected_level)
        
        passed = score_ok and level_ok
        if passed:
            passed_count += 1
            status_str = "PASS"
        else:
            status_str = "FAIL"
            outside_tolerance.append({
                "tc": case["tc"],
                "name": case["name"],
                "target": target_score,
                "actual": actual_score,
                "diff": diff,
                "tol": tol,
                "exp_level": expected_level,
                "act_level": actual_level
            })
            
        print(f"| {case['tc']:<6} | {case['name']:<32} | {target_score:<6} | {actual_score:<6} | {diff:<+5} | {expected_level:<10} | {actual_level:<10} | {status_str:<6} |")
        
    print("=" * 110)
    print(f"SUMMARY: {passed_count}/{total_count} PASSED ({(passed_count/total_count)*100:.1f}%)")
    if outside_tolerance:
        print(f"FAILED / OUTSIDE TOLERANCE CASES ({len(outside_tolerance)}):")
        for item in outside_tolerance:
            print(f"  - {item['tc']} ({item['name']}): Target={item['target']}, Actual={item['actual']} (Diff={item['diff']}, Tol=±{item['tol']}), Expected Level={item['exp_level']}, Actual Level={item['act_level']}")
    else:
        print("ALL BENCHMARK CASES PASSED WITHIN TOLERANCE AND EXACT LEVEL MATCH!")
    print("=" * 110)
    return passed_count == total_count

if __name__ == "__main__":
    run_benchmark()
