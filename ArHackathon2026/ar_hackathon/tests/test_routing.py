import os

from ar_hackathon.engine.game_engine import GameEngine
from ar_hackathon.api.routing import drive_unit_next_move


def test_level3_case_5_performs_well_under_station_congestion():
    test_case = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "test_cases",
        "level3",
        "test_case_5.json",
    )

    engine = GameEngine(test_case, drive_unit_next_move)
    stats = engine.run_until_finished()

    assert stats["delivered_pods"] >= 3
    assert stats["score"] >= 70
