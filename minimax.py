########################################################################################################
#   Description: Implemented minimax algorithm for a modified version of tic-tac-toe for my AI course.
#   This was a group project
#       - In this version, we have a 5 x 6 board, and the winning condition is getting 4 in a row
#       either horizontally, vertically, or diagonally
#       - We were told that there were 2 computer players, with player 1 (X) looking ahead 2 moves and
#       player 2 (O) looking 4 moves ahead (2 moves per player)
#       - Each player uses the minimax function to maximize the current player's chances of winning, 
#       assuming the other player will try to minimize the current player's chances
#       - The time for each move and the nodes generated before the move was made are provided in the 
#       output
########################################################################################################
import numpy as np
import time

#This is the MVP, future releases will make this configurable by the players
ROWS = 5
COLUMNS = 6
WINNING_LENGTH = 4

# Create an empty game board
game_board = np.full((ROWS, COLUMNS), ' ')

# Function to display the game board
def display_board(board):
    for row in board:
        print(" | ".join(row))
    print()

# Function to check if a move is valid - returns True/False
# A move is valid when the square is empty
def is_valid_move(board, row, col):
    return 0 <= row < ROWS and 0 <= col < COLUMNS and board[row][col] == ' '

# Function to make a move for a player
# A player will either be X or O
def make_move(board, row, col, player):
    board[row][col] = player

# Function to check for a win (4 in a row)
def check_win(board, player):
    # Check for horizontal wins
    for row in range(ROWS):
        for col in range(COLUMNS - WINNING_LENGTH + 1):
            if all(board[row][col + i] == player for i in range(WINNING_LENGTH)):
                return True

    # Check for vertical wins
    for col in range(COLUMNS):
        for row in range(ROWS - WINNING_LENGTH + 1):
            if all(board[row + i][col] == player for i in range(WINNING_LENGTH)):
                return True

    # Check for diagonal wins (both directions)
    for row in range(ROWS - WINNING_LENGTH + 1):
        for col in range(COLUMNS - WINNING_LENGTH + 1):
            if all(board[row + i][col + i] == player for i in range(WINNING_LENGTH)):
                return True
            if all(board[row + i][col + WINNING_LENGTH - 1 - i] == player for i in range(WINNING_LENGTH)):
                return True

    return False

# Evaluating the heuristic function based on the requirements given
# This function is used in minimax to help determine the best move
def heuristic_evaluation(board, player):
    # Initialize variables to count different patterns
    me = player
    opp_player = 'O' if player == 'X' else 'X'

    #This is the MVP, future releases could include different ways to configure these specific variables depending on the win condition
    #These variables help with the heuristic function at the end of this method

    #One side open 3 in a row means there's a 3 in a row, with an open side on one end
    opp_one_side_open_3_in_a_row = 0
    player_one_side_open_3_in_a_row = 0
    
    #Two side open 3 in a row is a 3 in a row with both sides being open
    opp_two_side_open_3_in_a_row = 0
    player_two_side_open_3_in_a_row = 0
    
    #Same as one side open 3 in a row but with a 2 in a row
    opp_one_side_open_2_in_a_row = 0
    player_one_side_open_2_in_a_row = 0
    
    #Same as two side open 3 in a row but with a 2 in a row
    opp_two_side_open_2_in_a_row = 0
    player_two_side_open_2_in_a_row = 0

    #Once again, this can also be made more generic in future releases, so this function could work with any condition
    # Check for 3-in-a-row and 2-in-a-row patterns
    for row in range(ROWS):
        for col in range(COLUMNS):
            #Checking for the patterns for the player (me)
            if board[row][col] == me:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                    count = 0
                    empty = 0
                    for i in range(-3, 4):
                        r, c = row + i * dr, col + i * dc
                        if 0 <= r < ROWS and 0 <= c < COLUMNS:
                            # If theres a match, increment count (number of Xs/Os) by 1 
                            if board[r][c] == me:
                                count += 1
                            # Else, if it's empty, increment the empty counter by 1
                            elif board[r][c] == ' ':
                                empty += 1
                    
                    #Incrementing variables as they appear, counting how many of these conditions we have
                    #We have a 3 in a row if count is 3
                    if count == 3:
                        if empty == 1:
                            player_one_side_open_3_in_a_row += 1
                        if empty == 2:
                            player_two_side_open_3_in_a_row += 1
                    #We have a 2 in a row if count is 2
                    if count == 2:
                        if empty == 1:
                            player_one_side_open_2_in_a_row += 1
                        if empty == 2:
                            player_two_side_open_2_in_a_row += 1
            
            #Doing the same thing for the opposing player
            elif board[row][col] == opp_player:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (-1, 1)]:
                    count = 0
                    empty = 0
                    for i in range(-3, 4):
                        r, c = row + i * dr, col + i * dc
                        if 0 <= r < ROWS and 0 <= c < COLUMNS:
                            if board[r][c] == me:
                                count += 1
                            elif board[r][c] == ' ':
                                empty += 1
                    if count == 3:
                        if empty == 1:
                            opp_one_side_open_3_in_a_row += 1
                        if empty == 2:
                            opp_two_side_open_3_in_a_row += 1
                    if count == 2:
                        if empty == 1:
                            opp_one_side_open_2_in_a_row += 1
                        if empty == 2:
                            opp_two_side_open_2_in_a_row += 1

    # This is the MVP, future releases will make this value more configurable, allowing the user to input their own
    # Calculate the heuristic value
    # This heuristic helps assign a value to a specific move which assists the minimax function and helps determine what move the player should make
    h = 200 * player_two_side_open_3_in_a_row - 80 * opp_two_side_open_3_in_a_row + \
        150 * player_one_side_open_3_in_a_row - 40 * opp_one_side_open_3_in_a_row + \
        20 * player_two_side_open_2_in_a_row - 15 * opp_two_side_open_2_in_a_row + \
        5 * player_one_side_open_2_in_a_row - 2 * opp_one_side_open_2_in_a_row

    return h

# NOTE: Because player 1 and player 2 have different depths (how many moves they look ahead), we made 2 separate functions to keep code simple
# Implement Player 1 (X)
def player1_minimax(board):
    # Display the initial game board
    display_board(board)

    # Run the minimax algorithm
    # This is the MVP, future releases will allow the depth (2 in the function call) to be configurable
    start_time = time.time()
    best_move, _ = minimax(board, 2, True, -np.inf, np.inf, 'X')
    end_time = time.time()
    move_time = end_time - start_time

    return best_move, move_time

# Implement Player 2 (O)
def player2_minimax(board):
    # Display the initial game board
    display_board(board)

    # Run the minimax algorithm
    # This is the MVP, future releases will allow the depth (4 in the function call) to be configurable
    start_time = time.time()
    best_move, _ = minimax(board, 4, True, -np.inf, np.inf, 'O')
    end_time = time.time()
    move_time = end_time - start_time

    return best_move, move_time

# Minimax algorithm with alpha-beta pruning
def minimax(board, depth, maximizing_player, alpha, beta, player):
    global nodes_generated  # Make nodes_generated global

    if depth == 0 or check_win(board, 'X') or check_win(board, 'O') or np.count_nonzero(board == ' ') == 0:
        if check_win(board, 'X'):
            return None, -1000
        elif check_win(board, 'O'):
            return None, 1000
        else:
            return None, heuristic_evaluation(board, player)

    if maximizing_player:
        max_eval = -np.inf
        best_move = None
        for row in range(ROWS):
            for col in range(COLUMNS):
                if is_valid_move(board, row, col):
                    nodes_generated += 1  # Increment node count
                    board[row][col] = player
                    _, eval = minimax(board, depth - 1, False, alpha, beta, player)
                    board[row][col] = ' '
                    if eval > max_eval:
                        max_eval = eval
                        best_move = (row, col)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return best_move, max_eval
    else:
        min_eval = np.inf
        best_move = None
        for row in range(ROWS):
            for col in range(COLUMNS):
                if is_valid_move(board, row, col):
                    nodes_generated += 1  # Increment node count
                    board[row][col] = player
                    _, eval = minimax(board, depth - 1, True, alpha, beta, player)
                    board[row][col] = ' '
                    if eval < min_eval:
                        min_eval = eval
                        best_move = (row, col)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return best_move, min_eval

# Play a game between Player 1 and Player 2
def play_game():
    global nodes_generated  # Reset nodes_generated for each game
    nodes_generated = 0
    
    # Make the initial starting move for player 1
    # Future releases could either have a random start position, or allow the start pposition to be configgurable
    game_board[2][3] = 'X'
    display_board(game_board)
    game_over = False
    # We start with player 2 (O) since player 1 made the first move already
    current_player = 'O'
    
    # This is how the game is being played, each player, on their turn, will call their respective minimax functions
    while not game_over:
        if current_player == 'X':
            print("Player 1 (X):")
            best_move, move_time = player1_minimax(game_board)
        else:
            print("Player 2 (O):")
            best_move, move_time = player2_minimax(game_board)

        # After the best move is calculated, make that move on the board and display the necessary information (time and nodes generated)
        if best_move is not None:
            row, col = best_move
            print(f"Player {current_player} moves to row {row}, column {col}")
            print(f"Nodes generated: {nodes_generated}")
            print(f"Move time: {move_time:.6f} seconds")

            make_move(game_board, row, col, current_player)
            display_board(game_board)

            # Check if someone wins and print the winner
            if check_win(game_board, current_player):
                print(f"Player {current_player} wins!")
                game_over = True

            # Check for a tie, future releases will have a gameover function to check a win and a tie simultaneously
            elif np.count_nonzero(game_board == ' ') == 0:
                print("It's a tie!")
                game_over = True
        else:
            print("Game over.")
            game_over = True

        # Switch players
        current_player = 'X' if current_player == 'O' else 'O'

# Initialize the number of nodes generated
nodes_generated = 0

# Start the game
play_game()
