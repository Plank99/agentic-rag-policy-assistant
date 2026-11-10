# Functional Evaluation Report

**Accuracy: 11/16 (68.8%)**

## Per-node latency (ms)

| node | count | avg_ms | max_ms |
|---|---|---|---|
| calculate_leave | 4 | 0.0 | 0.0 |
| classify_intent | 16 | 2860.0 | 4375.6 |
| create_ticket | 5 | 13.2 | 23.2 |
| direct_response | 1 | 3893.8 | 3893.8 |
| rag.generate_answer | 9 | 3911.9 | 6424.5 |
| rag.grade_relevance | 13 | 3359.9 | 4535.1 |
| rag.retrieve | 13 | 10.0 | 15.0 |
| rag.rewrite_query | 4 | 2807.2 | 4065.1 |
| rag_answer | 9 | 10030.0 | 19861.9 |
| synthesize_response | 15 | 6640.2 | 12956.6 |

## Per-question results

| id | type | pass | total_ms | detail |
|---|---|---|---|---|
| q1 | policy_question | PASS | 23633.3 | keywords found=['12 characters', 'multi-factor', 'MFA'] (looked for ['12 characters', 'multi-factor', 'MFA']) |
| q2 | policy_question | PASS | 10734.3 | keywords found=['5', 'carryover'] (looked for ['5', 'carryover']) |
| q3 | policy_question | PASS | 14018.5 | keywords found=['3 days'] (looked for ['3 days']) |
| q4 | policy_question | PASS | 15317.2 | keywords found=['25 EUR'] (looked for ['25 EUR']) |
| q5 | policy_question | FAIL | 29044.8 | keywords found=[] (looked for ['30 calendar days', '30 days']) |
| q6 | policy_question | PASS | 21506.4 | keywords found=['75 EUR'] (looked for ['75 EUR']) |
| q7 | leave_balance | PASS | 9226.6 | expected=37.0, actual=37.0 |
| q8 | leave_balance | PASS | 8858.2 | expected=10.5, actual=10.5 |
| q9 | leave_balance | PASS | 10278.7 | expected=-9.5, actual=-9.5 |
| q10 | create_ticket | FAIL | 6076.8 | expected_category=it_equipment_request, ticket={'ticket_id': 'TCK-0FFB58C1', 'category': 'it', 'summary': 'lost company laptop', 'priority': 'high', 'status': 'open', 'created_at': '2026-11-07T11:00:18.641861+00:00'} |
| q11 | create_ticket | FAIL | 15425.0 | expected_category=hr_conduct_concern, ticket={'ticket_id': 'TCK-FB844D6C', 'category': 'HR', 'summary': 'Confidential harassment concern', 'priority': 'high', 'status': 'open', 'created_at': '2026-11-07T11:00:24.736642+00:00'} |
| q12 | create_ticket | FAIL | 6461.2 | expected_category=finance_expense_dispute, ticket={'ticket_id': 'TCK-FBAC7536', 'category': 'expense_dispute', 'summary': 'Lost receipt for client dinner', 'priority': 'low', 'status': 'open', 'created_at': '2026-11-07T11:00:40.949114+00:00'} |
| q13 | compound | PASS | 12708.7 | keywords found=['5', 'carryover']; expected=8.5, actual=8.5 |
| q14 | compound | FAIL | 25870.6 | keywords found=['1 hour']; expected_category=it_access_request, ticket={'ticket_id': 'TCK-21FA622B', 'category': 'security_incident', 'summary': 'reporting security incident', 'priority': 'high', 'status': 'open', 'created_at': '2026-11-07T11:01:00.432639+00:00'} |
| q15 | out_of_scope | PASS | 5273.6 | routed to direct_response with no sub-task results, as expected |
| q16 | policy_question | PASS | 25199.7 | keywords found=['200 EUR'] (looked for ['200 EUR']) |