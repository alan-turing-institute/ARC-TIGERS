from arc_tigers.data.utils import clean_row


def test_clean_row():
    # Test with a sample row
    test_string = "This is a test sentence."
    test_string_len = len(test_string)
    row = {
        "text": "This is a test sentence.",
        "communityName": "community1",
        "len": 123,
        "other_field": None,
    }
    expected_row = {
        "text": test_string,
        "label": "community1",
        "len": test_string_len,
    }
    cleaned_row = clean_row(row)
    assert cleaned_row == expected_row

    # Test with a row that has no extra fields
    row = {
        "text": test_string,
        "communityName": "community2",
    }
    expected_row = {
        "text": test_string,
        "label": "community2",
        "len": test_string_len,
    }
    cleaned_row = clean_row(row)
    assert cleaned_row == expected_row
