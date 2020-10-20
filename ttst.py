import argparse
import itertools
import logging
import os
import pickle
import time

import numpy as np

EMPTY_BOARD = "000000000"

# Learning parameters
START_WEIGHT = 6
PENALTY = -1
REWARD = 3
DEFAULT_TRAINING = 5000

# User interface theme
CHARS = {"0": " ", "1": "O", "2": "X"}
logging.basicConfig(format="%(message)s")


# Command line interface
def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Play tic-tac-toe against a reinforcement learning agent."
            " If no brainfile is specified for a given player,"
            " then that player is assumed to be a human."
        )
    )
    parser.add_argument(
        "-p1", "--player1", metavar="P1", help="brainfile for the first player"
    )
    parser.add_argument(
        "-p2", "--player2", metavar="P2", help="brainfile for the second player"
    )
    parser.add_argument(
        "-t1", "--train1", metavar="P1", help="brainfile to train as a first player"
    )
    parser.add_argument(
        "-t2", "--train2", metavar="P2", help="brainfile to train as a second player"
    )
    parser.add_argument(
        "-n",
        "--iterations",
        metavar="INT",
        type=int,
        default=DEFAULT_TRAINING,
        help=f"how many matches to play during training, default is {DEFAULT_TRAINING}",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="don't print while training"
    )
    parser.add_argument(
        "-g", "--generate", metavar="FILE", help="write an empty brainfile to FILE"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite without asking when generating base brainfiles",
    )
    return parser


# Base brain generator
def brain_map():
    brain_map = {}
    for key in itertools.product("012", repeat=9):
        key = "".join(key)
        good_spots = [x == "0" for x in key]
        brain_map[key] = np.ones((9,), dtype=int) * good_spots
        brain_map[key] *= START_WEIGHT
    return brain_map


def write_brain_map(filename, force):
    if not force and os.path.isfile(filename):
        print(f"Brainfile {filename} already exists")
        answer = input("Overwrite? [y/N] ")
        if answer != "y":
            print("Quitting")
            return
    bmap = brain_map()
    with open(filename, "wb") as f:
        print(f"Saving base brain in {filename}")
        pickle.dump(bmap, f)
    return


# Input Output
def print_state(state):
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    print()
    print(" | ".join([CHARS[state[x]] for x in [6, 7, 8]]))
    print("---------")
    print(" | ".join([CHARS[state[x]] for x in [3, 4, 5]]))
    print("---------")
    print(" | ".join([CHARS[state[x]] for x in [0, 1, 2]]))
    print()


def print_tutorial():
    print()
    print("7 | 8 | 9 ")
    print("---------")
    print("4 | 5 | 6 ")
    print("---------")
    print("1 | 2 | 3 ")
    print()


def print_message_game_start(humans):
    players = []
    for human in humans:
        players.append("human" if human else "computer")
    print(f"Playing {players[0]} against {players[1]}")


def print_message_game_end(humans, result):
    name = CHARS[str(result)]
    logging.debug(f"{name} wins!")


def parse_move_linear(instr):
    try:
        x = int(instr)
    except ValueError:
        raise
    if x not in range(1, 10):
        raise ValueError
    return x - 1


def read_human_move(state):
    while True:
        move = input(">>> ")
        try:
            hmov = parse_move_linear(move)
        except ValueError:
            print("Invalid input, please enter number 1-9")
            continue
        if state[hmov] == "0":
            return hmov
        else:
            print("Cell already taken")
            continue


def load_brain(brainfile):
    try:
        with open(brainfile, "rb") as data:
            brainmap = pickle.load(data)
    except FileNotFoundError:
        logging.critical(f"File {brainfile} not found")
        raise
    except pickle.UnpicklingError:
        logging.critical(f"Malformed brainfile in {brainfile}")
        raise
    return brainmap


def update_brain(brain_map, moves, delta):
    for state, move in moves:
        # The following check should prevent death
        if delta > 0 or brain_map[state][move] > 1:
            brain_map[state][move] += delta


# Game handling
def game_result(state):
    for i in range(3):
        row = state[i * 3 : i * 3 + 3]
        if row == "111":
            return 1
        if row == "222":
            return 2
    for j in range(3):
        col = state[j : 7 + j : 3]
        if col == "111":
            return 1
        if col == "222":
            return 2
    main_diag = state[0] + state[4] + state[8]
    if main_diag == "111":
        return 1
    if main_diag == "222":
        return 2
    second_diag = state[2] + state[4] + state[6]
    if second_diag == "111":
        return 1
    if second_diag == "222":
        return 2
    if "0" not in state:
        assert state.count("2") == state.count("1") - 1
        return 2  # game is a tie: player 2 wins
    return 0


def cogitate(state, brain_map):
    distribution = brain_map[state] / sum(brain_map[state])
    return np.random.choice(9, 1, p=distribution)[0]


def make_human_move(player_id, state):
    hmov = read_human_move(state)
    assert state[hmov] == "0"
    updated = state[:hmov] + player_id + state[hmov + 1 :]
    return updated


def make_computer_move(player_id, brainmap, state):
    cmov = cogitate(state, brainmap)
    assert state[cmov] == "0"
    updated = state[:cmov] + player_id + state[cmov + 1 :]
    return cmov, updated


# Playing and training
def _mover_maker(player_id, human):
    def mover(state, bm):
        if human:
            newstate = make_human_move(player_id, state)
            return ([], newstate)
        else:
            return make_computer_move(player_id, bm, state)

    return mover


def play_core(bm1, bm2, t1, t2, n):
    humans = [x is None for x in [bm1, bm2]]
    print_message_game_start(humans)
    print_tutorial() if 1 in humans else None

    ids = ["1", "2"]
    bms = [bm1, bm2]
    movers = []
    for i in range(2):
        movers.append(_mover_maker(ids[i], humans[i]))

    def looper(matches, iterations):
        if 1 not in humans:
            return matches < iterations
        else:
            return matches < 1

    start = time.monotonic()
    matches = 0
    while looper(matches, n):
        state = EMPTY_BOARD
        moves = [[], []]
        result = 0
        while result == 0:
            for mover, record, bm in zip(movers, moves, bms):
                update = mover(state, bm)
                record.append((state, update[0]))
                state = update[1]
                print_state(state)
                result = game_result(state)
                if result != 0:
                    break

        print_message_game_end(humans, result)
        if result == 2:
            update_brain(bm1, moves[0], PENALTY) if t1 else None
            update_brain(bm2, moves[1], REWARD) if t2 else None
        else:
            assert result == 1
            update_brain(bm1, moves[0], REWARD) if t1 else None
            update_brain(bm2, moves[1], PENALTY) if t2 else None
        matches += 1

    end = time.monotonic()
    if 1 not in humans and (t1 == 1 or t2 == 1):
        logging.info(f"Trained {n} matches in {end - start:.3f} seconds")

    return bm1, bm2


def play(p1, p2, t1, t2, n):  # path path bool bool int
    try:
        brainmap1 = load_brain(p1) if p1 is not None else None
        brainmap2 = load_brain(p2) if p2 is not None else None
    except (FileNotFoundError, pickle.UnpicklingError):
        raise

    bm1, bm2 = play_core(brainmap1, brainmap2, t1, t2, n)

    if t1:
        with open(p1, "wb") as f:
            logging.info("Saving brain updates for computer 1")
            pickle.dump(bm1, f)
    if t2:
        with open(p2, "wb") as f:
            logging.info("Saving brain updates for computer 2")
            pickle.dump(bm2, f)


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.generate is not None:
        write_brain_map(args.generate, args.force)
        exit(0)

    if args.quiet:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)

    p1 = args.player1 or args.train1
    p2 = args.player2 or args.train2
    t1 = args.train1 is not None
    t2 = args.train2 is not None

    try:
        play(p1, p2, t1, t2, args.iterations)
    except (FileNotFoundError, pickle.UnpicklingError):
        exit(1)
