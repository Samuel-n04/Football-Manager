from tracking.config import (
    PASS_KICK_THRESH, PASS_ARRIVE_THRESH, BALL_PROX_PX, PASS_MAX_FLIGHT_FRAMES,
)
from tracking.team import closest_player_to_ball, same_team


def update_pass_detection(state, ball_speed, bcx, bcy, players_this_frame):
    near_pno, near_dist = closest_player_to_ball(bcx, bcy, players_this_frame)
    if state.pass_state == "idle":
        if ball_speed >= PASS_KICK_THRESH and state.last_ball_spd < PASS_KICK_THRESH:
            if near_dist <= BALL_PROX_PX and near_pno is not None:
                state.pass_from_pno        = near_pno
                state.pass_state           = "in_flight"
                state.pass_in_flight_since = state.frame_count
                state.passes_attempted[near_pno] += 1
    elif state.pass_state == "in_flight":
        # Timeout : si la balle n'arrive pas dans le délai imparti, annuler
        if state.frame_count - (state.pass_in_flight_since or state.frame_count) > PASS_MAX_FLIGHT_FRAMES:
            state.pass_state           = "idle"
            state.pass_from_pno        = None
            state.pass_in_flight_since = None
        elif ball_speed <= PASS_ARRIVE_THRESH:
            if (near_dist <= BALL_PROX_PX and near_pno is not None
                    and state.pass_from_pno is not None
                    and near_pno != state.pass_from_pno):
                if same_team(state, state.pass_from_pno, near_pno, players_this_frame):
                    state.passes_made[state.pass_from_pno] += 1
                    state.passes_received[near_pno]        += 1
            state.pass_state           = "idle"
            state.pass_from_pno        = None
            state.pass_in_flight_since = None
    state.last_ball_spd = ball_speed
