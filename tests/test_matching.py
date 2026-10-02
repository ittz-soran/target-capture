def test_part_of_plate_is_suggested():
    from targetcapture.plates import match_score

    assert match_score("21L1", "21L10415", 1) == (True, 0)
    assert match_score("104", "21 L 10415", 1)[0]
    assert not match_score("21", "21L10415", 1)[0]  # too short to suggest
