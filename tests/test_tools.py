from app.tools.calculator import compute_leave_balance
from app.tools.ticketing import create_ticket_record, get_ticket


def test_leave_balance_basic_accrual():
    result = compute_leave_balance(hire_date="2024-01-15", days_taken=3, as_of_date="2024-07-15")
    assert result["months_employed"] == 6
    assert result["accrued_days"] == 10.5
    assert result["remaining_days"] == 7.5


def test_leave_balance_partial_month_not_counted():
    # Hired on the 15th, "as of" the 10th of a later month: that partial month doesn't count.
    result = compute_leave_balance(hire_date="2024-01-15", days_taken=0, as_of_date="2024-03-10")
    assert result["months_employed"] == 1
    assert result["accrued_days"] == 1.75


def test_leave_balance_before_hire_date_is_zero():
    result = compute_leave_balance(hire_date="2024-06-01", days_taken=0, as_of_date="2024-01-01")
    assert result["months_employed"] == 0
    assert result["accrued_days"] == 0.0
    assert result["remaining_days"] == 0.0


def test_leave_balance_defaults_as_of_to_today():
    result = compute_leave_balance(hire_date="2020-01-01", days_taken=1)
    assert result["months_employed"] > 0


def test_create_and_fetch_ticket(tickets_db):
    ticket = create_ticket_record(
        category="it_access_request",
        summary="Need VPN access as a contractor",
        user_query="How do I get VPN access?",
        priority="high",
    )
    assert ticket["ticket_id"].startswith("TCK-")
    assert ticket["status"] == "open"

    fetched = get_ticket(ticket["ticket_id"])
    assert fetched is not None
    assert fetched["category"] == "it_access_request"
    assert fetched["priority"] == "high"


def test_create_ticket_invalid_priority_defaults_to_normal(tickets_db):
    ticket = create_ticket_record(category="hr_leave_request", summary="x", priority="urgent!!")
    assert ticket["priority"] == "normal"


def test_tickets_are_independent_rows(tickets_db):
    t1 = create_ticket_record(category="hr_leave_request", summary="a")
    t2 = create_ticket_record(category="hr_leave_request", summary="b")
    assert t1["ticket_id"] != t2["ticket_id"]
