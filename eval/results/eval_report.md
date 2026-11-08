# Functional Evaluation Report

**Accuracy: 12/16 (75.0%)**

## Per-node latency (ms)

| node | count | avg_ms | max_ms |
|---|---|---|---|
| calculate_leave | 4 | 0.0 | 0.0 |
| classify_intent | 16 | 3568.5 | 45398.0 |
| create_ticket | 4 | 14.2 | 20.6 |
| direct_response | 1 | 933.5 | 933.5 |
| rag.generate_answer | 9 | 889.8 | 1658.4 |
| rag.grade_relevance | 9 | 413.4 | 499.6 |
| rag.retrieve | 9 | 118.1 | 989.3 |
| rag_answer | 9 | 1423.5 | 3073.6 |
| synthesize_response | 15 | 1111.0 | 1621.8 |

## Per-question results

| id | type | pass | total_ms | detail |
|---|---|---|---|---|
| q1 | policy_question | PASS | 50073.5 | keywords found=['12 characters', 'multi-factor', 'MFA'] (looked for ['12 characters', 'multi-factor', 'MFA']) |
| q2 | policy_question | PASS | 2515.6 | keywords found=['5'] (looked for ['5', 'carryover']) |
| q3 | policy_question | PASS | 2630.6 | keywords found=['3 days'] (looked for ['3 days']) |
| q4 | policy_question | PASS | 2610.7 | keywords found=['25 EUR'] (looked for ['25 EUR']) |
| q5 | policy_question | PASS | 2578.6 | keywords found=['30 calendar days'] (looked for ['30 calendar days', '30 days']) |
| q6 | policy_question | PASS | 2776.5 | keywords found=['75 EUR'] (looked for ['75 EUR']) |
| q7 | leave_balance | PASS | 2295.1 | expected=37.0, actual=37.0 |
| q8 | leave_balance | PASS | 2548.9 | expected=10.5, actual=10.5 |
| q9 | leave_balance | PASS | 2584.5 | expected=-9.5, actual=-9.5 |
| q10 | create_ticket | FAIL | 2210.3 | expected_category=it_equipment_request, ticket={'ticket_id': 'TCK-52E1B5B4', 'category': 'it', 'summary': 'Lost company laptop while traveling', 'priority': 'high', 'status': 'open', 'created_at': '2026-11-09T14:51:03.275308+00:00'} |
| q11 | create_ticket | FAIL | 2129.1 | expected_category=hr_conduct_concern, ticket={'ticket_id': 'TCK-F4293751', 'category': 'confidential_report', 'summary': 'Harassment concern about a coworker', 'priority': 'high', 'status': 'open', 'created_at': '2026-11-09T14:51:05.550233+00:00'} |
| q12 | create_ticket | FAIL | 1855.1 | expected_category=finance_expense_dispute, ticket={'ticket_id': 'TCK-A8E17929', 'category': 'expense-dispute', 'summary': 'Lost receipt for 150 EUR client dinner', 'priority': 'normal', 'status': 'open', 'created_at': '2026-11-09T14:51:07.665699+00:00'} |
| q13 | compound | PASS | 3258.5 | keywords found=['5']; expected=8.5, actual=8.5 |
| q14 | compound | FAIL | 3909.6 | keywords found=['1 hour']; expected_category=it_access_request, ticket={'ticket_id': 'TCK-51486D6D', 'category': 'vpn_access', 'summary': "Request for new contractor's VPN access", 'priority': 'normal', 'status': 'open', 'created_at': '2026-11-09T14:51:14.396727+00:00'} |
| q15 | out_of_scope | PASS | 1260.8 | routed to direct_response with no sub-task results, as expected |
| q16 | policy_question | PASS | 2398.4 | keywords found=['200 EUR'] (looked for ['200 EUR']) |