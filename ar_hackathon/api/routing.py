"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: Underconfidence Interval
Email address: sarahphu648@gmail.com, sara168818@gmail.com, lotuswei@outlook.com
*******************
"""

import heapq
from typing import Dict, List, Optional, Tuple

from ar_hackathon.models.graph_state import GraphState


_UNIT_ASSIGNMENTS: Dict[int, str] = {}
_STUCK_STATE: Dict[int, Tuple[int, int, int]] = {}
_LAST_TIME_STEP: Optional[int] = None


def pathing(state: GraphState, start: int, goal: int) -> Optional[Tuple[float, List[int]]]:
    """Return the weighted shortest path from start to goal using Dijkstra."""
    if start == goal:
        return 0.0, [start]

    open_set = [(0.0, 0.0, start)]
    costs: Dict[int, float] = {start: 0.0}
    previous: Dict[int, int] = {}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if cost != costs.get(current):
            continue
        if current == goal:
            path = [current]
            while current in previous:
                current = previous[current]
                path.append(current)
            path.reverse()
            return cost, path

        for neighbor in state.neighbors(current):
            edge = state.get_edge(current, neighbor)
            if edge is None:
                continue
            new_cost = cost + edge.weight
            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                previous[neighbor] = current
                heapq.heappush(open_set, (new_cost, new_cost, neighbor))

    return None


def _traffic_aware_path(
    state: GraphState, start: int, goal: int
) -> Optional[Tuple[float, List[int]]]:
    """Find a route that does not knowingly enter a full edge or node."""
    if start == goal:
        return 0.0, [start]

    open_set = [(0.0, start)]
    costs: Dict[int, float] = {start: 0.0}
    previous: Dict[int, int] = {}

    while open_set:
        cost, current = heapq.heappop(open_set)
        if cost != costs.get(current):
            continue
        if current == goal:
            route = [current]
            while current in previous:
                current = previous[current]
                route.append(current)
            route.reverse()
            return cost, route

        for neighbor in state.neighbors(current):
            edge = state.get_edge(current, neighbor)
            if edge is None:
                continue
            if edge.capacity is not None and state.edge_occupancy(current, neighbor) >= edge.capacity:
                continue
            node = state.get_node(neighbor)
            if (
                neighbor != goal
                and node is not None
                and node.capacity is not None
                and state.node_occupancy(neighbor) >= node.capacity
            ):
                continue

            new_cost = cost + edge.weight
            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                previous[neighbor] = current
                heapq.heappush(open_set, (new_cost, neighbor))

    return None


def _record_stuck(unit_id: int, state: GraphState) -> None:
    """Print a one-time diagnostic when a unit has not changed position."""
    unit = state.get_drive_unit(unit_id)
    if unit is None or unit.in_transit:
        _STUCK_STATE.pop(unit_id, None)
        return

    last_node, count, last_step = _STUCK_STATE.get(unit_id, (unit.current_node, 0, -1))
    if last_node == unit.current_node and last_step != state.current_time_step:
        count += 1
    else:
        count = 1
    _STUCK_STATE[unit_id] = (unit.current_node, count, state.current_time_step)
    if count == 5:
        print(
            f"routing debug: unit {unit_id} unchanged at node "
            f"{unit.current_node} for 5 calls (step {state.current_time_step})"
        )


def _clear_assignments(state: GraphState) -> None:
    """Drop stale pod assignments left behind by old or completed work."""
    active_pod_ids = {pod.id for pod in state.active_pods}
    for unit_id, pod_id in list(_UNIT_ASSIGNMENTS.items()):
        unit = state.get_drive_unit(unit_id)
        if unit is None:
            del _UNIT_ASSIGNMENTS[unit_id]
            continue
        pod = state.get_pod(pod_id)
        if pod is None or pod.id not in active_pod_ids or pod.carried_by is not None:
            del _UNIT_ASSIGNMENTS[unit_id]


def _station_penalty(state: GraphState, destination_station: int, unit_id: int) -> int:
    """Prefer less-congested stations and avoid duplicate station assignments."""
    penalty = 0
    for other_unit in state.drive_units:
        if other_unit.id == unit_id:
            continue
        for carried_pod_id in other_unit.carrying:
            carried_pod = state.get_pod(carried_pod_id)
            if carried_pod is not None and carried_pod.destination_station == destination_station:
                penalty += 100
        if other_unit.in_transit and other_unit.transit_destination == destination_station:
            penalty += 50
        elif not other_unit.in_transit and other_unit.current_node == destination_station:
            penalty += 50

    for assigned_unit_id, pod_id in _UNIT_ASSIGNMENTS.items():
        if assigned_unit_id == unit_id:
            continue
        other_pod = state.get_pod(pod_id)
        if other_pod is not None and other_pod.destination_station == destination_station:
            penalty += 200

    node = state.get_node(destination_station)
    if node is not None and node.capacity is not None:
        if state.node_occupancy(destination_station) >= node.capacity:
            penalty += 500
    return penalty


def _carried_route(state: GraphState, unit_id: int) -> Optional[Tuple[float, List[int]]]:
    """Choose the quickest currently usable destination among carried pods."""
    unit = state.get_drive_unit(unit_id)
    if unit is None:
        return None

    choices = []
    for pod_id in unit.carrying:
        pod = state.get_pod(pod_id)
        if pod is None:
            continue
        station = state.get_node(pod.destination_station)
        full = (
            station is not None
            and station.capacity is not None
            and state.node_occupancy(station.id) >= station.capacity
        )
        route = _traffic_aware_path(state, unit.current_node, pod.destination_station)
        if route is None:
            continue
        choices.append((full, route[0], pod.entry_time, pod.id, route))

    if not choices:
        return None
    choices.sort(key=lambda choice: (choice[0], choice[1], choice[2], choice[3]))
    return choices[0][4]


def _yield_move(state: GraphState, unit_id: int) -> Optional[int]:
    """Move an idle unit to a nearby, unoccupied node so it does not park in traffic."""
    unit = state.get_drive_unit(unit_id)
    if unit is None:
        return None

    candidates = []
    for neighbor in state.neighbors(unit.current_node):
        edge = state.get_edge(unit.current_node, neighbor)
        if edge is None:
            continue
        if edge.capacity is not None and state.edge_occupancy(unit.current_node, neighbor) >= edge.capacity:
            continue
        node = state.get_node(neighbor)
        if node is not None and node.capacity is not None:
            if state.node_occupancy(neighbor) >= node.capacity:
                continue
        station_penalty = 1000 if node is not None and node.node_type == "station" else 0
        candidates.append((station_penalty + state.node_occupancy(neighbor), edge.weight, neighbor))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    This is the function that students will implement. The game engine will
    call this function for each idle drive unit at each time step to
    determine where it should go next.

    Pickups and deliveries are automatic: a drive unit with free capacity
    that stops at (or passes through) a node with a waiting pod picks it up,
    and a drive unit that reaches a carried pod's destination station drops
    it off.

    Args:
        drive_unit_id: ID of the drive unit being routed
        state: GraphState object containing the current state of the floor

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
                      at the current node
    """
    global _LAST_TIME_STEP

    if state.current_time_step == 0 and _LAST_TIME_STEP != 0:
        _UNIT_ASSIGNMENTS.clear()
        _STUCK_STATE.clear()
    _LAST_TIME_STEP = state.current_time_step

    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    _record_stuck(drive_unit_id, state)
    _clear_assignments(state)

    if unit.carrying:
        route = _carried_route(state, unit.id)
        if route is None or len(route[1]) < 2:
            return _yield_move(state, unit.id)
        destination = route[1][-1]
        station = state.get_node(destination)
        if (
            station is not None
            and station.capacity is not None
            and state.node_occupancy(station.id) >= station.capacity
            and station.id in state.neighbors(unit.current_node)
        ):
            return None
        return route[1][1]

    if not unit.has_capacity:
        return None

    assigned_pod_id = _UNIT_ASSIGNMENTS.get(unit.id)
    if assigned_pod_id is not None:
        pod = state.get_pod(assigned_pod_id)
        if pod is not None and pod.current_node is not None and pod.carried_by is None:
            route = _traffic_aware_path(state, unit.current_node, pod.current_node)
            if route is not None and len(route[1]) >= 2:
                return route[1][1]
        del _UNIT_ASSIGNMENTS[unit.id]

    best_choice = None
    best_score = None
    for pod in sorted(state.active_pods, key=lambda p: (p.entry_time, p.id)):
        if pod.carried_by is not None or pod.current_node is None:
            continue
        if any(other_pod_id == pod.id for other_pod_id in _UNIT_ASSIGNMENTS.values()):
            continue

        route_to_pod = _traffic_aware_path(state, unit.current_node, pod.current_node)
        if route_to_pod is None:
            continue
        route_to_station = _traffic_aware_path(state, pod.current_node, pod.destination_station)
        if route_to_station is None:
            continue

        score = (
            route_to_pod[0] + route_to_station[0]
            + _station_penalty(state, pod.destination_station, unit.id)
            - 2.0 * min(state.current_time_step - pod.entry_time, 10),
            route_to_pod[0],
            pod.entry_time,
            pod.id,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_choice = (route_to_pod[1], pod.id)

    if best_choice is None:
        return _yield_move(state, unit.id)

    path_to_pod, pod_id = best_choice
    _UNIT_ASSIGNMENTS[unit.id] = pod_id
    if len(path_to_pod) < 2:
        return None
    return path_to_pod[1]
