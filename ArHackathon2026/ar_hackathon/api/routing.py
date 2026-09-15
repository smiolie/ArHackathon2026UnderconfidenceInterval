"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: 
Email address: sarahphu648@gmail.com
*******************
"""

import heapq
from typing import Dict, List, Optional, Tuple
from ar_hackathon.models.graph_state import GraphState


def _a_star_path(state: GraphState, start: int, goal: int) -> Optional[Tuple[float, List[int]]]:
    """Return the weighted shortest path from start to goal using A*."""
    if start == goal:
        return 0, [start]

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
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    targets = []
    if unit.carrying:
        for pod_id in unit.carrying:
            pod = state.get_pod(pod_id)
            if pod is not None:
                targets.append((pod.destination_station, pod.entry_time, pod.id))
    elif unit.has_capacity:
        for pod in state.active_pods:
            if pod.carried_by is None and pod.current_node is not None:
                targets.append((pod.current_node, pod.entry_time, pod.id))

    best_route = None
    for target, entry_time, pod_id in targets:
        route = _a_star_path(state, unit.current_node, target)
        if route is None:
            continue
        route_cost, path = route
        candidate = (route_cost, entry_time, pod_id, path)
        if best_route is None or candidate[:3] < best_route[:3]:
            best_route = candidate

    if best_route is None or len(best_route[3]) < 2:
        return None
    return best_route[3][1]
