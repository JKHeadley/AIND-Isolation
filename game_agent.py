"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""

from random import randint
import numpy


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def check_partition(game, player):
    """Check if a partition exists on the board. We only check for partitions
     where two side-by-side rows or columns are completely filled. If a partition
     exists, we conclude that the player on the side with the most open squares wins.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        inf: win
        -inf: loss
        0: none
    """

    exists = False
    rows = ()
    columns = ()
    
    own_loc = ()
    own_side = 0
    
    opp_loc = ()
    opp_side = 0
    
    # First check row partitions

    own_loc = game.get_player_location(player)
    opp_loc = game.get_player_location(game.get_opponent(player))
    for i in range(2, 4):
        for j in range(0, 7):
            if game.move_is_legal((i, j)) or game.move_is_legal((i+1, j)):
                break
            elif j == 6:
                own_loc = game.get_player_location(player)
                opp_loc = game.get_player_location(game.get_opponent(player))
                # players cant be inside the partition
                print(game.to_string())
                if own_loc[0] != i and own_loc[0] != i + 1 and opp_loc[0] != i and opp_loc[0] != i + 1:
                    exists = True
                    print(game.to_string())
        if exists:
            rows = (i, i+1)
            break
        
    # If a partition exists, see if players are on opposite sides (-1 is top, +1 is bottom)
    if exists:
        own_loc = game.get_player_location(player)
        if own_loc[0] <= rows[0]:
            own_side = -1
        else:
            own_side = 1

        opp_loc = game.get_player_location(game.get_opponent(player))
        if opp_loc[0] <= rows[0]:
            opp_side = -1
        else:
            opp_side = 1

        # If players are on opposite sides, we approximate that the winner is on the larger side
        # NOTE: A more accurate (but more costly) estimate would be to count open moves available
        # on each side.
        if own_side != opp_side:
            if rows[0] < 3 and opp_side == -1:
                return float("inf")
            else:
                return float("-inf")

    own_loc = game.get_player_location(player)
    opp_loc = game.get_player_location(game.get_opponent(player))
    for j in range(2, 4):
        for i in range(0, 7):
            if game.move_is_legal((i, j)) or game.move_is_legal((i, j + 1)):
                break
            elif i == 6:
                own_loc = game.get_player_location(player)
                opp_loc = game.get_player_location(game.get_opponent(player))
                # players cant be inside the partition
                print(game.to_string())
                if own_loc[1] != j and own_loc[1] != j + 1 and opp_loc[1] != j and opp_loc[1] != i + j:
                    exists = True
                    print(game.to_string())
        if exists:
            columns = (j, j + 1)
            break

    return 0


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # if game.move_count > 18:
    #     partition = check_partition(game, player)
    #     if partition == float("inf") or partition == float("-inf"):
    #         return partition

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def custom_score2(game, player):
    """
    A heuristic that favors the center of the board for the player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    a = numpy.array(game.get_player_location(player))
    b = numpy.array((3, 3))
    dist = -1 * numpy.linalg.norm(a - b)
    return float(dist)


def custom_score3(game, player):
    """
    A heuristic that favors the center of the board for the player and the edges of the
    board for the opponent.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    a = numpy.array(game.get_player_location(player))
    b = numpy.array(game.get_player_location(game.get_opponent(player)))
    c = numpy.array((3, 3))
    my_dist = -1 * numpy.linalg.norm(a - c)
    opp_dist = numpy.linalg.norm(b - c)

    return float(my_dist + opp_dist)


def custom_score4(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    a = numpy.array(game.get_player_location(player))
    b = numpy.array(game.get_player_location(game.get_opponent(player)))
    c = numpy.array((3, 3))
    my_dist = -1 * numpy.linalg.norm(a - c)
    opp_dist = numpy.linalg.norm(b - c)
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float((my_dist + opp_dist) * (own_moves - opp_moves))


def custom_score5(game, player):
    """

    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opp_loc = game.get_player_location(game.get_opponent(player))
    if opp_loc in [(0, 0), (0, 6), (6, 0), (6, 6)]:
        return float("inf")

    own_loc = game.get_player_location(player)
    if own_loc in [(0, 0), (0, 6), (6, 0), (6, 6)]:
        return float("-inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def custom_score6(game, player):
    """
    A modified "improved" heuristic.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2 * opp_moves)


def custom_score7(game, player):
    """
    A heuristic that tries to keep the opponent close
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    a = numpy.array(game.get_player_location(player))
    b = numpy.array(game.get_player_location(game.get_opponent(player)))
    dist = -1 * numpy.linalg.norm(a - b)
    return float(dist)


def custom_score8(game, player):
    """
    A heuristic that tries to keep the opponent far
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    a = numpy.array(game.get_player_location(player))
    b = numpy.array(game.get_player_location(game.get_opponent(player)))
    dist = numpy.linalg.norm(a - b)
    return float(dist)


def custom_score9(game, player):
    """
    A heuristic that changes with move_count
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - game.move_count * opp_moves)


def custom_score10(game, player):
    """
    A heuristic that changes with move_count (alternative)
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(game.move_count * own_moves - opp_moves)


def custom_score11(game, player):
    """
    A heuristic that changes with move_count (alternative). Coefficients should be determined through
    a genetic algorithm.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves**(player.own_coef * game.move_count) - opp_moves**(player.opp_coef * game.move_count))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=110., name="", own_coef=1, opp_coef=1):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.isOpponent = False
        self.ab_trees = dict()
        self.move_count = 0
        self.name = name
        self.reflect = False
        self.center = ()
        self.last_opponent_location = ()
        self.own_coef = own_coef
        self.opp_coef = opp_coef

        self.depth_at_move = dict()

        self.branching_factor = dict()

    def get_best_second_move(self, game):
        move = tuple(map(lambda x, y: x - y, self.center, (0, 1)))
        return move

    def get_reflect_move(self, game):
        if game.get_player_location(self) == self.center:
            translation = tuple(map(lambda x, y: x - y, game.get_player_location(game.get_opponent(self)), self.center))
            move = tuple(map(lambda x, y: x - y, self.center, translation))
        else:
            translation = tuple(map(lambda x, y: x - y, self.last_opponent_location, game.get_player_location(game.get_opponent(self))))
            move = tuple(map(lambda x, y: x + y, game.get_player_location(self), translation))

        self.last_opponent_location = game.get_player_location(game.get_opponent(self))
        return move

    def can_reflect(self, game):
        if game.get_player_location(self) == self.center:
            reflect_moves = {
                tuple(map(sum, zip(self.center, (-2, -1)))),
                tuple(map(sum, zip(self.center, (-2, 1)))),
                tuple(map(sum, zip(self.center, (-1, -2)))),
                tuple(map(sum, zip(self.center, (-1, 2)))),
                tuple(map(sum, zip(self.center, (1, -2)))),
                tuple(map(sum, zip(self.center, (1, 2)))),
                tuple(map(sum, zip(self.center, (2, -1)))),
                tuple(map(sum, zip(self.center, (2, 1))))
            }
            if game.get_player_location(game.get_opponent(self)) in reflect_moves:
                self.reflect = True
                return True

        return False

    def get_opening_move(self, game):
        move = (-1, -1)
        self.center = (int(game.width / 2), int(game.height / 2))
        if game.move_count == 0:
            move = self.center
            self.reflect = False
        elif game.move_count == 1:
            self.reflect = False
            if game.move_is_legal(self.center):
                move = self.center
            else:
                move = self.get_best_second_move(game)
        elif game.move_count == 2:
            if self.can_reflect(game):
                move = self.get_reflect_move(game)

        if move in game.get_legal_moves(self):
            return move
        else:
            return None

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        value = {}
        move = {}
        last_depth = False
        lookout = False

        self.ab_trees = dict()

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        game.show_moves()
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if game.move_count < 3:
                move = self.get_opening_move(game)
                if not move:
                    move = game.get_legal_moves()[randint(0, len(game.get_legal_moves()) - 1)]
            elif self.reflect:
                move = self.get_reflect_move(game)
                # print("REFLECTING", move, game.get_legal_moves(), move in game.get_legal_moves(self))
                # new_game = game.forecast_move(move)
                # print(new_game.to_string())
                if not move in game.get_legal_moves(self) and len(game.get_legal_moves()) > 0:
                    self.reflect = False
                    move = game.get_legal_moves()[randint(0, len(game.get_legal_moves()) - 1)]
                    print("REFLECTION FAILED:", move, game.get_legal_moves(), move in game.get_legal_moves(self))
                    print(game.to_string())

            elif self.iterative:
                for depth in range(1, 100):
                    last_depth = depth
                    # After ~move 28, the average branching factor is 2 and AB pruning isn't effective
                    if game.move_count < 28:
                        value, move = getattr(self, self.method)(game, depth)
                    else:
                        value, move = self.minimax(game, depth)
                    # The following if statements end iterative deepening early based on the prediction of a win or loss
                    if value == float("-inf") and depth > 50:
                        break
                    if value == float("inf"):
                        break
            else:
                value, move = getattr(self, self.method)(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        # if self.iterative and last_depth > 12:
            # print("STUDENT MOVE", move)
            # print("STUDENT VALUE", value)
            # print("STUDENT MOVES", game.move_count)
            # print("LAST DEPTH: ", last_depth)
            # print(game.to_string())
        # else:
        #     print("OPPONENT MOVE", move)
        #     print("OPPONENT VALUE", value)
        #     print("OPPONENT DEPTH", last_depth)
        # print("STUDENT MOVE", move, value)
        if self.iterative and last_depth:
            self.depth_at_move[game.move_count] = last_depth

        self.move_count += 1
        return move

    def minimax(self, game, depth, maximizing_player=True, first=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            value = self.score(game, self), game.get_player_location(game.active_player)
            # if game.active_player.iterative:
            #     print("END VALUE OPPONENT: ", value)
            # else:
            #     print("END VALUE STUDENT: ", value)
            return value

        moves = dict()

        if maximizing_player:
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                # print(new_game.visited)
                # print(new_game.counter)
                moves[move], _ = self.minimax(new_game, depth - 1, False, False)
                # if first:
                #     print(new_game.to_string())
                #     print(depth, move, moves[move])

            if len(moves) > 0:
                value = max(moves, key=moves.get)
                # if first:
                #     print(moves)
                #     print(value)
                return moves[value], value
            else:
                return float("-inf"), (-1, -1)
        else:
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                # print(new_game.visited)
                # print(new_game.counter)
                moves[move], _ = self.minimax(new_game, depth - 1, True, False)
                # if first:
                #     print(new_game.to_string())
                #     print(depth, move, moves[move])

            if len(moves) > 0:
                value = min(moves, key=moves.get)
                # if first:
                #     print(moves)
                #     print(value)
                return moves[value], value
            else:
                return float("inf"), (-1, -1)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, first=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        current_location = game.get_player_location(game.active_player)

        if depth == 0:
            return self.score(game, self), current_location

        moves = dict()

        # Empirical data shows that after around move 18, optimal ordering of branches is not worth the effort (gives worse results)
        if game.move_count < 18:
            if current_location in self.ab_trees:
                legal_moves = (k[0] for k in self.ab_trees[current_location])
            else:
                legal_moves = game.get_legal_moves()
        else:
            legal_moves = game.get_legal_moves()

        if maximizing_player:
            for move in legal_moves:
                new_game = game.forecast_move(move)
                v, _ = self.alphabeta(new_game, depth - 1, alpha, beta, False)
                moves[move] = v
                # print(new_game.to_string())
                # print(depth, move, moves[move])
                if v >= beta:
                    return v, move
                alpha = max([alpha, v])

            if game.move_count < 18:
                self.ab_trees[current_location] = sorted(moves.items(), key=lambda x: x[1],
                                                     reverse=True)

            if len(moves) > 0:
                value = max(moves, key=moves.get)
                return moves[value], value
            else:
                return float("-inf"), (-1, -1)
        else:
            for move in legal_moves:
                new_game = game.forecast_move(move)
                v, _ = self.alphabeta(new_game, depth - 1, alpha, beta, True)
                moves[move] = v
                # print(new_game.to_string())
                # print(depth, move, moves[move])
                if v <= alpha:
                    return v, move
                beta = min([beta, v])

            if game.move_count < 18:
                self.ab_trees[current_location] = sorted(moves.items(), key=lambda x: x[1],
                                                     reverse=False)

            if len(moves) > 0:
                value = min(moves, key=moves.get)
                return moves[value], value
            else:
                return float("inf"), (-1, -1)


class CustomPlayerOpponent:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=110., name=""):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.isOpponent = True
        self.move_count = 0
        self.name = name

        self.depth_at_move = dict()

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        value = {}
        move = {}
        last_depth = {}

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if game.move_count == 0:
                return legal_moves[randint(0, len(legal_moves) - 1)]
            elif game.move_count == 1:
                return legal_moves[randint(0, len(legal_moves) - 1)]

            if self.iterative:
                for depth in range(1, 100):
                    last_depth = depth
                    value, move = getattr(self, self.method)(game, depth)
                    if value == float("-inf") and depth > 50:
                        break
                    if value == float("inf"):
                        break
            else:
                value, move = getattr(self, self.method)(game, self.search_depth)


        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        # if self.iterative and last_depth > 7:
        #     print("STUDENT MOVE", move)
        #     print("STUDENT VALUE", value)
        #     print("OPPONENT DEPTH", last_depth)
        #     print("GAME MOVES", game.move_count)
        # else:
        #     print("OPPONENT MOVE", move)
        #     print("OPPONENT VALUE", value)
        #     print("OPPONENT DEPTH", last_depth)
        if self.iterative:
            self.depth_at_move[game.move_count] = last_depth

        self.move_count += 1
        return move

    def minimax(self, game, depth, maximizing_player=True, first=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            value = self.score(game, self), game.get_player_location(game.active_player)
            # if game.active_player.iterative:
            #     print("END VALUE OPPONENT: ", value)
            # else:
            #     print("END VALUE STUDENT: ", value)
            return value

        moves = dict()

        if maximizing_player:
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                # print(new_game.visited)
                # print(new_game.counter)
                moves[move], _ = self.minimax(new_game, depth - 1, False, False)
                # if first:
                #     print(new_game.to_string())
                #     print(depth, move, moves[move])

            if len(moves) > 0:
                value = max(moves, key=moves.get)
                # if first:
                #     print(moves)
                #     print(value)
                return moves[value], value
            else:
                return float("-inf"), (-1, -1)
        else:
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                # print(new_game.visited)
                # print(new_game.counter)
                moves[move], _ = self.minimax(new_game, depth - 1, True, False)
                # if first:
                #     print(new_game.to_string())
                #     print(depth, move, moves[move])

            if len(moves) > 0:
                value = min(moves, key=moves.get)
                # if first:
                #     print(moves)
                #     print(value)
                return moves[value], value
            else:
                return float("inf"), (-1, -1)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), game.get_player_location(game.active_player)

        moves = dict()

        if maximizing_player:
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                v, _ = self.alphabeta(new_game, depth - 1, alpha, beta, False)
                moves[move] = v
                if v >= beta:
                    return v, move
                alpha = max([alpha, v])

            if len(moves) > 0:
                value = max(moves, key=moves.get)
                return moves[value], value
            else:
                return float("-inf"), (-1, -1)
        else:
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                v, _ = self.alphabeta(new_game, depth - 1, alpha, beta, True)
                moves[move] = v
                if v <= alpha:
                    return v, move
                beta = min([beta, v])

            if len(moves) > 0:
                value = min(moves, key=moves.get)
                return moves[value], value
            else:
                return float("inf"), (-1, -1)
