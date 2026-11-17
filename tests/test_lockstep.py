import pytest

from net.lockstep import (
    fixed,
    fixed_mul,
    sort_inputs,
    detect_desync,
    room_seed,
    advance_frame,
    state_hash,
    accept_input
)


def test_fixed_positive():
    assert fixed(1.5) == 1500

def test_fixed_negative():
    assert fixed(-0.001) == -1

def test_fixed_mul():
    assert fixed_mul(1500, 2000) == 3000

def test_fixed_mul_saturates():
    assert fixed_mul(2**31 - 1, 2) == 2**31 - 1

def test_sort_by_frame():
    assert [i['frame'] for i in sort_inputs([{'frame': 2, 'player': 1}, {'frame': 1, 'player': 2}])] == [1, 2]

def test_sort_same_frame_by_player():
    assert sort_inputs([{'frame': 1, 'player': 2}, {'frame': 1, 'player': 1}]) == [{'frame': 1, 'player': 1}, {'frame': 1, 'player': 2}]

def test_desync_detects():
    assert detect_desync([1, 1, 2]) == [2]

def test_desync_none():
    assert detect_desync([1, 1, 1]) == []

def test_seed_deterministic():
    assert room_seed('room-a') == room_seed('room-a')

def test_seed_differs():
    assert room_seed('room-a') != room_seed('room-b')

def test_advance_frame():
    assert advance_frame(5) == 6

def test_state_hash_same():
    assert state_hash({'x': 1}, []) == state_hash({'x': 1}, [])

def test_state_hash_diff():
    assert state_hash({'x': 1}, []) != state_hash({'x': 2}, [])

def test_sort_empty():
    assert sort_inputs([]) == []

def test_accept_input_window():
    assert accept_input(1, 2) is False and accept_input(2, 2) is True
