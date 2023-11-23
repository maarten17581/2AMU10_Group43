#!/usr/bin/env python3

#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple
from competitive_sudoku.execute import execute_command


# Prints 1 instead of 1.0
def print_score(x: float) -> str:
    return '0' if x == 0 else str(x).rstrip('0').rstrip('.')


# Play a game between the two players, and return the result.
def play_game(player: str, opponent: str, board: str, calculation_time: float, output_file: str) -> Tuple[float, float]:
    cmd = f'python3 ./simulate_game.py --first={player} --second={opponent} --board={board} --time={calculation_time}'
    output = execute_command(cmd)
    Path(output_file).write_text(output)
    if 'Player 1 wins the game.' in output:
        return (1, 0)
    elif 'The game ends in a draw.' in output:
        return (0.5, 0.5)
    elif 'Player 2 wins the game.' in output:
        return (0, 1)
    else:
        print(output)
        raise RuntimeError('Unknown game result')


# Play a match between player and opponent.
def play_match(player: str, opponent: str, count: int, board: str, calculation_time: float) -> None:
    player_score = 0.0
    opponent_score = 0.0
    result_lines = []

    for i in range(1, count+1):
        print(f'Playing game {i}')
        player_starts = i % 2 == 1
        first = player if player_starts else opponent
        second = opponent if player_starts else player
        output_file = f'{player}-{opponent}-game={i}-board={Path(board).stem}-time={calculation_time}.txt'
        result = play_game(first, second, board, calculation_time, output_file)

        result_line = f'{first} - {second} {print_score(result[0])}-{print_score(result[1])}\n'
        result_lines.append(result_line)
        print(result_line)

        if player_starts:
            player_score += result[0]
            opponent_score += result[1]
        else:
            player_score += result[1]
            opponent_score += result[0]

    result_line = f'Match result: {player} - {opponent} {print_score(player_score)}-{print_score(opponent_score)}'
    result_lines.append(result_line)
    print(result_line)

    output_file = f'{player}-{opponent}-board={Path(board).stem}-time={calculation_time}-match-result.txt'
    Path(output_file).write_text('\n'.join(result_lines))


def main():
    cmdline_parser = argparse.ArgumentParser(description='Play a match between two sudoku players.')
    cmdline_parser.add_argument('--first', help="The module name of the first player's SudokuAI class (default: random_player)", default='random_player')
    cmdline_parser.add_argument('--second', help="The module name of the second player's SudokuAI class (default: random_player)", default='random_player')
    cmdline_parser.add_argument('--count', type=int, default=6, help='The number of games (default: 6)')
    cmdline_parser.add_argument('--board', type=str, default='boards/empty-2x2.txt', help='The text file containing the start position (default: boards/empty-2x2.txt)')
    cmdline_parser.add_argument('--time', type=float, default=3.0, help="The time (in seconds) for computing a move (default: 3.0)")
    args = cmdline_parser.parse_args()

    play_match(args.first, args.second, args.count, args.board, args.time)


if __name__ == '__main__':
    main()
