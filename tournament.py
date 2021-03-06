"""
Estimate the strength rating of student-agent with iterative deepening and
a custom heuristic evaluation function against fixed-depth minimax and
alpha-beta search agents by running a round-robin tournament for the student
agent. Note that all agents are constructed from the student CustomPlayer
implementation, so any errors present in that class will affect the outcome
here.

The student agent plays a fixed number of "fair" matches against each test
agent. The matches are fair because the board is initialized randomly for both
players, and the players play each match twice -- switching the player order
between games. This helps to correct for imbalances in the game due to both
starting position and initiative.

For example, if the random moves chosen for initialization are (5, 2) and
(1, 3), then the first match will place agentA at (5, 2) as player 1 and
agentB at (1, 3) as player 2 then play to conclusion; the agents swap
initiative in the second match with agentB at (5, 2) as player 1 and agentA at
(1, 3) as player 2.
"""

from profilers import *
import itertools
import random
import warnings

from collections import namedtuple

from isolation import Board
from sample_players import RandomPlayer
from sample_players import HumanPlayer
from sample_players import null_score
from sample_players import open_move_score
from sample_players import improved_score
from game_agent import CustomPlayer
from game_agent import CustomPlayerMC
from game_agent import MonteCarlo
from game_agent import CustomPlayerOpponent
from game_agent import CustomPlayerOpponent2
from game_agent import CustomPlayerOpponent3
from game_agent import custom_score
from game_agent import custom_score2
from game_agent import custom_score3
from game_agent import custom_score4
from game_agent import custom_score5
from game_agent import custom_score6
from game_agent import custom_score7
from game_agent import custom_score8
from game_agent import custom_score9
from game_agent import custom_score10
from game_agent import custom_score11
from game_agent import custom_score12
from game_agent import custom_score13
from game_agent import custom_score14
from game_agent import custom_score15

NUM_MATCHES = 200  # number of matches against each opponent
TIME_LIMIT = 250  # number of milliseconds before timeout
GENETIC = False

TIMEOUT_WARNING = "One or more agents lost a match this round due to " + \
                  "timeout. The get_move() function must return before " + \
                  "time_left() reaches 0 ms. You will need to leave some " + \
                  "time for the function to return, and may need to " + \
                  "increase this margin to avoid timeouts during  " + \
                  "tournament play."

DESCRIPTION = """
This script evaluates the performance of the custom heuristic function by
comparing the strength of an agent using iterative deepening (ID) search with
alpha-beta pruning against the strength rating of agents using other heuristic
functions.  The `ID_Improved` agent provides a baseline by measuring the
performance of a basic agent using Iterative Deepening and the "improved"
heuristic (from lecture) on your hardware.  The `Student` agent then measures
the performance of Iterative Deepening and the custom heuristic against the
same opponents.
"""

Agent = namedtuple("Agent", ["player", "name"])

first = True
branching_factor = dict()
match_count = dict()
win_count = dict()
avg_depth_at_move = dict()
avg_time = dict()
reflection_wins = dict()
count_reached = dict()


def play_match(player1, player2):
    """
    Play a "fair" set of matches between two agents by playing two games
    between the players, forcing each agent to play from randomly selected
    positions. This should control for differences in outcome resulting from
    advantage due to starting position on the board.
    """

    global first, branching_factor, match_count, avg_depth_at_move
    num_wins = {player1: 0, player2: 0}
    num_timeouts = {player1: 0, player2: 0}
    num_invalid_moves = {player1: 0, player2: 0}
    games = [Board(player1, player2), Board(player2, player1)]
    # games = [Board(player2, player1), Board(player2, player1)]

    # initialize both games with a random move and response
    # for _ in range(2):
    #     move = random.choice(games[0].get_legal_moves())
    #     games[0].apply_move(move)
    #     games[1].apply_move(move)

    # play both games and tally the results
    for game in games:
        winner, _, termination = game.play(time_limit=TIME_LIMIT)

        if player1 == winner:
            # if game == games[0]:
            #     if not player1.isOpponent:
            #         print("WINNER: ", "PLAYER 1: ", winner.name)
            # else:
            #     if not player1.isOpponent:
            #         print("WINNER: ", "PLAYER 2: ", winner.name)


            num_wins[player1] += 1

            if termination == "timeout":
                print("TIMEOUT")
                num_timeouts[player2] += 1
            else:
                num_invalid_moves[player2] += 1

        elif player2 == winner:
            # if game == games[0]:
            #     if not player2.isOpponent:
            #         print("WINNER: ", "PLAYER 2: ", winner.name)
            # else:
            #     if not player2.isOpponent:
            #         print("WINNER: ", "PLAYER 1: ", winner.name)

            num_wins[player2] += 1

            if termination == "timeout":
                print("TIMEOUT:")
                num_timeouts[player1] += 1
            else:
                num_invalid_moves[player1] += 1

        # The code below is used to calculate the average branching factor as a function of the number of moves made
        # if player1.isOpponent:
        #     branching_factor = {k: player2.branching_factor.get(k, 0) + branching_factor.get(k, 0) for k in
        #                          set(player2.branching_factor) | set(branching_factor)}
        # else:
        #     branching_factor = {k: player1.branching_factor.get(k, 0) + branching_factor.get(k, 0) for k in
        #                         set(player1.branching_factor) | set(branching_factor)}
        #

        # print(branching_factor)

        if player1.name in match_count:
            match_count[player1.name] += 1
        else:
            match_count[player1.name] = 1

        if player2.name in match_count:
            match_count[player2.name] += 1
        else:
            match_count[player2.name] = 1

        # if player1 in avg_time:
        #     avg_time[player1] = game.time_left[player1] + avg_time[player1]
        # else:
        #     avg_time[player1] = game.time_left[player1]
        #
        # if player2 in avg_time:
        #     avg_time[player2] = game.time_left[player2] + avg_time[player2]
        # else:
        #     avg_time[player2] = game.time_left[player2]

        # if player1.name in avg_depth_at_move:
        #     avg_depth_at_move[player1.name] = {k: player1.depth_at_move.get(k, 0) + avg_depth_at_move[player1.name].get(k, 0) for k in set(player1.depth_at_move) | set(avg_depth_at_move[player1.name])}
        # else:
        #     avg_depth_at_move[player1.name] = player1.depth_at_move
        #
        # if player2.name in avg_depth_at_move:
        #     avg_depth_at_move[player2.name] = {k: player2.depth_at_move.get(k, 0) + avg_depth_at_move[player2.name].get(k, 0) for k in set(player2.depth_at_move) | set(avg_depth_at_move[player2.name])}
        # else:
        #     avg_depth_at_move[player2.name] = player2.depth_at_move

        # When calculating avg_depth, we only take into account games that were won, since losses can result in irrelevant deep searches
        if winner.name in avg_depth_at_move:
            avg_depth_at_move[winner.name] = {
            k: winner.depth_at_move.get(k, 0) + avg_depth_at_move[winner.name].get(k, 0) for k in
            set(winner.depth_at_move) | set(avg_depth_at_move[winner.name])}
        else:
            avg_depth_at_move[winner.name] = winner.depth_at_move

        if winner.name in win_count:
            win_count[winner.name] += 1
        else:
            win_count[winner.name] = 1

        if winner.name in count_reached:
            for i in range(0, len(winner.depth_at_move) + 1):
                if i in count_reached[winner.name]:
                    count_reached[winner.name][i] += 1
                else:
                    count_reached[winner.name][i] = 1
        else:
            for i in range(0, len(winner.depth_at_move) + 1):
                count_reached[winner.name] = dict()
                count_reached[winner.name][i] = 1
            
        # print(game.to_string())
        # print("WINNER: ", winner.name)
        # if winner.name != "Student":
        #     print(game.to_string())
        if winner.name == "Student" or winner.name == "Student2" or winner.name == "Student3" or winner.name == "Student4" and winner.reflect:
            if winner.name in reflection_wins:
                reflection_wins[winner.name] += 1
            else:
                reflection_wins[winner.name] = 1

    if sum(num_timeouts.values()) != 0:
        warnings.warn(TIMEOUT_WARNING)

    return num_wins[player1], num_wins[player2]


# @do_profile(follow=[MonteCarlo.get_play, MonteCarlo.run_simulation])
def play_round(agents, num_matches):
    """
    Play one round (i.e., a single match between each pair of opponents)
    """
    global branching_factor, match_count
    agent_1 = agents[-1]
    wins = 0.
    total = 0.

    print("\nPlaying Matches:")
    print("----------")

    for idx, agent_2 in enumerate(agents[:-1]):

        if agent_1.player != agent_2.player:
            counts = {agent_1.player: 0., agent_2.player: 0.}
            names = [agent_1.name, agent_2.name]
            print("  Match {}: {!s:^11} vs {!s:^11}".format(idx + 1, *names), end=' ')

            # Each player takes a turn going first
            for p1, p2 in itertools.permutations((agent_1.player, agent_2.player)):
                for _ in range(num_matches):
                    score_1, score_2 = play_match(p1, p2)
                    counts[p1] += score_1
                    counts[p2] += score_2
                    total += score_1 + score_2

            wins += counts[agent_1.player]

            # print("  Match {}: {!s:^11} vs {!s:^11}".format(idx + 1, *names), end=' ')
            print("\tResult: {} to {}".format(int(counts[agent_1.player]),
                                              int(counts[agent_2.player])))

    # branching_factor = {k: v / match_count for k, v in branching_factor.items()}

    # avg_time[agent_1.player] = avg_time[agent_1.player] / match_count
    # avg_time[agent_2.player] = avg_time[agent_2.player] / match_count

    # print(agent_1.name, "AVERAGE TIME: ", avg_time[agent_1.player])
    # print(agent_2.name, "AVERAGE TIME: ", avg_time[agent_2.player])

    return 100. * wins / total


def main():

    HEURISTICS = [("Null", null_score),
                  ("Open", open_move_score),
                  ("Improved", improved_score)]
    # HEURISTICS = [("Null", null_score)]
    MM_ARGS = {"search_depth": 3, "method": 'minimax', "iterative": False}
    AB_ARGS = {"search_depth": 5, "method": 'alphabeta', "iterative": False}
    CUSTOM_ARGS = {"method": 'alphabeta', 'iterative': True}
    CUSTOM_ARGS_MM = {"method": 'minimax', 'iterative': True}

    # Create a collection of CPU agents using fixed-depth minimax or alpha beta
    # search, or random selection.  The agent names encode the search method
    # (MM=minimax, AB=alpha-beta) and the heuristic function (Null=null_score,
    # Open=open_move_score, Improved=improved_score). For example, MM_Open is
    # an agent using minimax search with the open moves heuristic.
    mm_agents = [Agent(CustomPlayerOpponent(score_fn=h, **MM_ARGS, name="MM_" + name),
                       "MM_" + name) for name, h in HEURISTICS]
    ab_agents = [Agent(CustomPlayerOpponent(score_fn=h, **AB_ARGS, name="AB_" + name),
                       "AB_" + name) for name, h in HEURISTICS]
    random_agents = [Agent(RandomPlayer(), "Random")]

    best_agents = [Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS, name="ID_Improved_Optimized"), "ID_Improved_Optimized"),
                   Agent(CustomPlayer(score_fn=custom_score3, **CUSTOM_ARGS, name="Student3"), "Student3   "),
                   Agent(CustomPlayer(score_fn=custom_score8, **CUSTOM_ARGS, name="Student8"), "Student8   "),
                   Agent(CustomPlayer(score_fn=custom_score13, **CUSTOM_ARGS, name="Student13", own_coef=1.492220782479327, opp_coef=0.7729218598739231), "Student13   "),
                   Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="Student15", own_coef=0.4, opp_coef=1.7, own_const=0.3, opp_const=0.5, modifier=40), "Student15   ")]

    human_agent = [Agent(HumanPlayer(), "Human")]

    # ID_Improved agent is used for comparison to the performance of the
    # submitted agent for calibration on the performance across different
    # systems; i.e., the performance of the student agent is considered
    # relative to the performance of the ID_Improved agent to account for
    # faster or slower computers.

    # test_agents = [Agent(CustomPlayerOpponent(score_fn=improved_score, **CUSTOM_ARGS, name="ID_Improved"), "ID_Improved"),
    #                Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS, name="ID_Improved_Optimized"), "ID_Improved_Optimized"),
    #                Agent(CustomPlayer(score_fn=custom_score12, **CUSTOM_ARGS, name="Student12", own_coef=1.492220782479327, opp_coef=0.7729218598739231), "Student12   "),
    #                Agent(CustomPlayer(score_fn=custom_score14, **CUSTOM_ARGS, name="Student14", own_coef=0.8483477579717855, opp_coef=0.6863555382980071), "Student14   "),
    #                Agent(CustomPlayer(score_fn=custom_score3, **CUSTOM_ARGS, name="Student3"), "Student3   ")]

    # test_agents = [Agent(CustomPlayer(score_fn=custom_score12, **CUSTOM_ARGS, name="0", own_coef=1.492220782479327, opp_coef=0.7729218598739231), "0   "),
    #                Agent(CustomPlayer(score_fn=custom_score12, **CUSTOM_ARGS, name="1", own_coef=1.5618571223729099, opp_coef=0.8217416162055202), "1   "),
    #                Agent(CustomPlayer(score_fn=custom_score12, **CUSTOM_ARGS, name="2", own_coef=1.511722601561318, opp_coef=0.8796045168998665), "2   "),
    #                Agent(CustomPlayer(score_fn=custom_score12, **CUSTOM_ARGS, name="3", own_coef=1.1006980138853246, opp_coef=0.9458528727213513), "3   "),
    #                Agent(CustomPlayer(score_fn=custom_score12, **CUSTOM_ARGS, name="4", own_coef=1.1365094743620907, opp_coef=0.9571847560010487), "4   ")]


    # test_agents = [Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="15",
    #                                   own_coef=0.4, opp_coef=1.7, own_const=0.3,
    #                                   opp_const=0.5, modifier=40), "15   "),
    #                Agent(CustomPlayer(score_fn=custom_score13, **CUSTOM_ARGS, name="Student13-1",
    #                                   own_coef=0.85, opp_coef=1), "Student13-1   "),
    #                Agent(CustomPlayer(score_fn=custom_score13, **CUSTOM_ARGS, name="Student13-2",
    #                                   own_coef=1.492220782479327, opp_coef=0.7729218598739231), "Student13-2   "),
    #                Agent(CustomPlayer(score_fn=custom_score12, **CUSTOM_ARGS, name="Student12", own_coef=1.492220782479327, opp_coef=0.7729218598739231), "Student12   ")]


    # test_agents = [
    #                Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="1",
    #                                   own_coef=0.362188268462442, opp_coef=1.504392575883522, own_const=0.45030953991436634,
    #                                   opp_const=0.3915814204702087, modifier=53.80620285705746), "1   "),
    #                Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="2",
    #                                   own_coef=0.4438438125899901, opp_coef=1.971980054483135, own_const=1.160099913496322,
    #                                   opp_const=1.3932233794237263, modifier=10.204529023045676), "2   "),
    #                 Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="0",
    #                                    own_coef=0.4, opp_coef=1.7, own_const=0.3,
    #                                    opp_const=0.5, modifier=40), "0   "),
    #                Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="3",
    #                                   own_coef=0.7138401082059999, opp_coef=0.9156694399426819, own_const=0.5952675035858037,
    #                                   opp_const=0.21402322408404145, modifier=33.6577841046563), "3   "),
    #                Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="4",
    #                                   own_coef=0.42284197796430584, opp_coef=1.4739239632218115, own_const=0.2798712658943987,
    #                                   opp_const=0.4855681530065824, modifier=35.46485388071125), "4   "),
    #                Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="5",
    #                                   own_coef=0.3414446564809855, opp_coef=1.7126342279499907, own_const=0.4043076172052206,
    #                                   opp_const=0.5095011853080299, modifier=41.37358711324541), "5   "),
    #                ]

    test_agents = [
        Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="1",
                           own_coef=1.7044629891425531, opp_coef=1.6490487183260913, own_const=0.7894738206437453,
                           opp_const=0.4491943511988748, modifier=1.4192648079546637), "1   "),
        Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name="0",
                           own_coef=0.4, opp_coef=1.7, own_const=0.3,
                           opp_const=0.5, modifier=40), "0   ")
    ]


    # test_agents = [Agent(CustomPlayerOpponent(score_fn=improved_score, **CUSTOM_ARGS_MM, name="ID_Improved_MM"), "ID_Improved_MM"),
    #                Agent(CustomPlayerOpponent(score_fn=improved_score, **CUSTOM_ARGS, name="ID_Improved_AB"), "ID_Improved_AB"),
    #                Agent(CustomPlayerOpponent(score_fn=improved_score, **CUSTOM_ARGS, name="ID_Improved_Dynamic", dynamic=True), "ID_Improved_Dynamic")]

    # test_agents = [Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS_MM, name="ID_Improved_MM"), "ID_Improved_MM"),
    #                Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS, name="ID_Improved_AB"), "ID_Improved_AB"),
    #                Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS, name="ID_Improved_Dynamic", dynamic=True),
    #           "ID_Improved_Dynamic")]

    # test_agents = [Agent(CustomPlayer(score_fn=custom_score7, **CUSTOM_ARGS, name="Student7"), "Student7   "),
    #                Agent(CustomPlayer(score_fn=custom_score8, **CUSTOM_ARGS, name="Student8"), "Student8   ")]

    # test_agents = [Agent(CustomPlayerOpponent(score_fn=improved_score, **CUSTOM_ARGS_MM, name="ID_Improved"), "ID_Improved"),
    #                Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS, name="Student"), "Student    ")]

    # test_agents = [Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS, name="Student"), "Student    ")]

    # test_agents = [Agent(CustomPlayerMC(score_fn=improved_score, name="MC"), "MC   ")]

    # test_agents = [Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS, name="AB0"), "AB0    "),
    #                Agent(CustomPlayerOpponent(score_fn=improved_score, **CUSTOM_ARGS, name="AB1"), "AB1    "),
    #                Agent(CustomPlayerOpponent2(score_fn=improved_score, **CUSTOM_ARGS, name="AB2"), "AB2    "),
    #                Agent(CustomPlayerOpponent3(score_fn=improved_score, **CUSTOM_ARGS, name="AB3"), "AB3    ")]

    # test_agents = [Agent(CustomPlayerOpponent3(score_fn=improved_score, **CUSTOM_ARGS, name="AB2"), "AB2    ")]

    # test_agents = [Agent(CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS, name="ID_Improved"), "ID_Improved")]

    if GENETIC:
        initial_coef = [
            (random.uniform(0, 2), random.uniform(0, 2)),
            (random.uniform(0, 2), random.uniform(0, 2)),
            (random.uniform(0, 2), random.uniform(0, 2)),
            (random.uniform(0, 2), random.uniform(0, 2))
        ]

        initial_const = [
            (random.uniform(0, 2), random.uniform(0, 2)),
            (random.uniform(0, 2), random.uniform(0, 2)),
            (random.uniform(0, 2), random.uniform(0, 2)),
            (random.uniform(0, 2), random.uniform(0, 2))
        ]

        initial_modifier = [
            random.uniform(0, 4) * 10,
            random.uniform(0, 4) * 10,
            random.uniform(0, 4) * 10,
            random.uniform(0, 4) * 10,
        ]

        # initial_coef = [
        #     (0.4, 1.7),
        #     (random.uniform(0, 2), random.uniform(0, 2)),
        #     (random.uniform(0, 2), random.uniform(0, 2)),
        #     (random.uniform(0, 2), random.uniform(0, 2))
        # ]
        #
        # initial_const = [
        #     (0.3, 0.5),
        #     (random.uniform(0, 2), random.uniform(0, 2)),
        #     (random.uniform(0, 2), random.uniform(0, 2)),
        #     (random.uniform(0, 2), random.uniform(0, 2))
        # ]
        #
        # initial_modifier = [
        #     40,
        #     random.uniform(0, 4) * 10,
        #     random.uniform(0, 4) * 10,
        #     random.uniform(0, 4) * 10,
        # ]

        # initial_coef = [
        #     (0.4, 1.7),
        #     (0.39028435408560014, 1.6500267921210408),
        #     (0.40824903981871197, 1.2921657974552068)
        # ]
        #
        # initial_const = [
        #     (0.3, 0.5),
        #     (0.4409356902471761, 0.4769695014241872),
        #     (0.9270522605215425, 0.769269982864594)
        # ]
        #
        # initial_modifier = [
        #     40,
        #     4.757091027264442,
        #     0.7911139015621447,
        # ]

        best = [((0, 0), (0, 0), 0, 0)]
        # initial_values = []

        performance = []
        test_agents = []

        # for i in range(0, 3):
        #     initial_values.append((random.uniform(-1, 2), random.uniform(-1, 2)))

        #test_agents.append(Agent(CustomPlayer(score_fn=custom_score12, **CUSTOM_ARGS, name=-1, own_coef=best[0][0][0], opp_coef=best[0][0][1]), -1))
        # initialize agents with random weights
        for i in range(0, 4):
            # initial_values.append((random.uniform(-1, 2), random.uniform(-1, 2)))
            # test_agents.append(Agent(CustomPlayer(score_fn=custom_score14, **CUSTOM_ARGS, name=i, own_coef=best[i][0][0], opp_coef=best[i][0][1]), i))
            test_agents.append(Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name=i, own_coef=initial_coef[i][0], opp_coef=initial_coef[i][1],
                                     own_const=initial_const[i][0], opp_const=initial_const[i][1], modifier=initial_modifier[i]), i))

        print(DESCRIPTION)

        print("INITIAL COEF: " + str(initial_coef) + "\n")
        print("INITIAL CONST: " + str(initial_const) + "\n")
        print("INITIAL MOD: " + str(initial_modifier) + "\n")

        while True:
            for agentUT in test_agents:

                agents = best_agents + [agentUT]

                win_ratio = play_round(agents, NUM_MATCHES)

                performance.append(((agentUT.player.own_coef, agentUT.player.opp_coef), (agentUT.player.own_const, agentUT.player.opp_const), agentUT.player.modifier, win_ratio))

                print(str(performance[-1][3]) + "\n")

            performance = sorted(performance, key=lambda x: x[3], reverse=True)

            local_best = performance[0]

            current_best = best[-1]

            if local_best[-1] > current_best[-1]:
                print("FOUND BETTER: " + str(local_best) + "\n")
                best.append(local_best)

            current_best_coefs = best[-1][0]
            current_best_consts = best[-1][1]
            current_best_modifier = best[-1][2]
            second_best_coefs = performance[1][0]
            second_best_consts = performance[1][1]
            second_best_modifier = performance[1][2]

            coefs = [
                (current_best_coefs[0] + random.uniform(-0.1, 0.1), current_best_coefs[1] + random.uniform(-0.1, 0.1)),
                (current_best_coefs[0] + random.uniform(-0.2, 0.2), current_best_coefs[1] + random.uniform(-0.2, 0.2)),
                (second_best_coefs[0] + random.uniform(-0.3, 0.3), second_best_coefs[1] + random.uniform(-0.3, 0.3)),
                (random.uniform(0, 2), random.uniform(0, 2))
            ]

            consts = [
                (current_best_consts[0] + random.uniform(-0.1, 0.1), current_best_consts[1] + random.uniform(-0.1, 0.1)),
                (current_best_consts[0] + random.uniform(-0.2, 0.2), current_best_consts[1] + random.uniform(-0.2, 0.2)),
                (second_best_consts[0] + random.uniform(-0.3, 0.3), second_best_consts[1] + random.uniform(-0.3, 0.3)),
                (random.uniform(0, 2), random.uniform(0, 2))
            ]

            modifiers = [
                current_best_modifier + random.uniform(-5, 5),
                current_best_modifier + random.uniform(-10, 10),
                second_best_modifier + random.uniform(-15, 15),
                random.uniform(0, 3.5) * 10
            ]

            print("BEST:", best)

            print("NEW COEFS: " + str(coefs) + "\n")
            print("NEW CONSTS: " + str(consts) + "\n")
            print("NEW MODS: " + str(modifiers) + "\n")

            for i in range(0, 4):
                test_agents[i] = Agent(CustomPlayer(score_fn=custom_score15, **CUSTOM_ARGS, name=i, own_coef=coefs[i][0],
                                                    opp_coef=coefs[i][1], own_const=consts[i][0], opp_const=consts[i][1], modifier=modifiers[i]), i)

    else:

        print(DESCRIPTION)
        for agentUT in test_agents:
            print("")
            print("*************************")
            print("{:^25}".format("Evaluating: " + agentUT.name))
            print("*************************")

            # agents = random_agents + mm_agents + ab_agents + [agentUT]
            # agents = random_agents + mm_agents + [agentUT]
            # agents = random_agents + [agentUT]
            # agents = mm_agents + [agentUT]
            # agents = ab_agents + [agentUT]
            # agents = human_agent + [agentUT]
            # agents = [Agent(CustomPlayerOpponent(score_fn=custom_score, **CUSTOM_ARGS), "Opponent")] + [agentUT]
            # agents = [Agent(CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS, name="Opponent"), "Opponent")] + [agentUT]
            agents = best_agents + [agentUT]
            # agents = test_agents + [agentUT]
            win_ratio = play_round(agents, NUM_MATCHES)

            print("\n\nResults:")
            print("----------")
            print("{!s:<15}{:>10.2f}%".format(agentUT.name, win_ratio))

        for agent in test_agents:
            avg_depth_at_move[agent.player.name] = {k: v / win_count[agent.player.name] for k, v in
                                                    avg_depth_at_move[agent.player.name].items()}
            for k, v in avg_depth_at_move[agent.player.name].items():
                v = round(v, 2)
                avg_depth_at_move[agent.player.name][k] = v

            count_reached[agent.player.name] = sorted(count_reached[agent.player.name].items(), key=lambda x: x[0],
                                                      reverse=False)

            print("COUNTS REACHED FOR ", agent.name, count_reached[agent.player.name])

        for agent in test_agents:
            if agent.player.name == "Student" or agent.player.name == "Student2" or agent.player.name == "Student3" or agent.player.name == "Student4":
                print("REFLECTION WINS:", agent.name, reflection_wins[agent.player.name])

if __name__ == "__main__":
    main()
