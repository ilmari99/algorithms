# Tic-Tac-Toe Program using
# random number in Python
 
# importing all necessary libraries
import os
import numpy as np
import random
from time import sleep
import tensorflow as tf
import matplotlib.pyplot as plt
from convert_tf_to_tflite import convert_to_tflite
#MODEL = tf.keras.models.load_model('model.h5')

class LiteModel:
    def __init__(self, path, expand_dims = False):
        if path == "random":
            self.predict = lambda x: [[random.random()]]
            return
        self.expand_dims = expand_dims
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()
        
    def predict(self, inp):
        if self.expand_dims:
            inp = np.expand_dims(inp,axis=-1)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
    




# Creates an empty board
 
 
def create_board():
    return(np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]))
 
# Check for empty places on board
 
 
def possibilities(board):
    l = []
 
    for i in range(len(board)):
        for j in range(len(board)):
 
            if board[i][j] == 0:
                l.append((i, j))
    return(l)
 
# Select a random place for the player
 
 
def random_place(board, player):
    selection = possibilities(board)
    current_loc = random.choice(selection)
    board[current_loc] = player
    return(board)

def human_place(board,player):
    print("Current board: ")
    selection = possibilities(board)
    print("Possible choices are: {}".format(selection))
    choice = input("Enter your choice: ")
    choice = tuple(map(int,choice.split(",")))
    if choice in selection:
        board[choice] = player
    else:
        print("Invalid choice")
        return human_place(board,player)
    return board

def get_prediction(pl,board):
    inp = np.array([[pl] + [pl] + list(board.flatten())],dtype=np.float32)
    pred = MODEL.predict(inp)
    return pred[0][0]

def model_place(board,player, p = 0):
    selection = possibilities(board)
    choice_evaluations = []
    for choice in selection:
        board[choice] = player
        pred = get_prediction(player, board)
        choice_evaluations.append((choice,pred))
        #print(board)
        #print("Prediction: {}".format(pred))
        board[choice] = 0
    #print(f"Choices: {choice_evaluations}")
    choice_evaluations = sorted(choice_evaluations, key=lambda x: x[1], reverse=True)
    # Use weighted top-p sampling
    evals_exp = np.exp([x[1] for x in choice_evaluations])
    evals = evals_exp/np.sum(evals_exp)
    #print(f"Weights: {evals}")
    cumsum = np.cumsum(evals)
    # Find the first index where the cumulative sum is greater than p
    idx = np.argmax(cumsum >= p)
    #print(f"Index: {idx}")
    valid_choices = choice_evaluations[:idx+1]
    # Re softmax the valid choices to get the choice probabilities
    evals_exp = np.exp([x[1] for x in valid_choices])
    evals = evals_exp/np.sum(evals_exp)
    # Choose a random move from the top-p choices
    #print("Valid choices: {}".format(valid_choices))
    #print(f"Weights: {evals}")
    indices = np.arange(len(valid_choices))
    choice_idx = np.random.choice(indices, p=evals)
    choice = valid_choices[choice_idx][0]
    #print(f"Chose: {choice}")
    board[choice] = player
    #print("Vector: {}".format([[player] + list(board.flatten())]))
    return board

def model_place_and_random(board,player, rate = 0.25):
    if random.random() < rate:
        return random_place(board,player)
    return model_place(board,player)
# Checks whether the player has three
# of their marks in a horizontal row
 
 
def row_win(board, player):
    for x in range(len(board)):
        win = True
 
        for y in range(len(board)):
            if board[x, y] != player:
                win = False
                continue
 
        if win == True:
            return(win)
    return(win)
 
# Checks whether the player has three
# of their marks in a vertical row
 
 
def col_win(board, player):
    for x in range(len(board)):
        win = True
 
        for y in range(len(board)):
            if board[y][x] != player:
                win = False
                continue
 
        if win == True:
            return(win)
    return(win)
 
# Checks whether the player has three
# of their marks in a diagonal row
 
 
def diag_win(board, player):
    win = True
    y = 0
    for x in range(len(board)):
        if board[x, x] != player:
            win = False
    if win:
        return win
    win = True
    if win:
        for x in range(len(board)):
            y = len(board) - 1 - x
            if board[x, y] != player:
                win = False
    return win
 
# Evaluates whether there is
# a winner or a tie
 
 
def evaluate(board):
    winner = 0
 
    for player in [-1,1]:
        if (row_win(board, player) or
                col_win(board, player) or
                diag_win(board, player)):
 
            winner = player
 
    if np.all(board != 0) and winner == 0:
        winner = 0.5
    return winner


 
 
# Main function to start the game
def play_game(players = {"p1":random_place,"p2":random_place},rate = 0.25, gather_data = False, shuffle = True, verbose = False):
    board, winner, counter = create_board(), 0, 1
    pl_states = {"p1":[], "p2":[]}
    player_ids = {"p1":1, "p2":-1}
    if shuffle and random.random() > 0.5:
        player_ids = {"p1":-1, "p2":1}
        players = {"p2":players["p2"],"p1":players["p1"]}
    if verbose:
        print(f"Player 1 is {player_ids['p1']} and Player 2 is {player_ids['p2']}")
    while winner == 0:
        for player,fun in players.items():
            i = player_ids[player]
            args = [board,i]
            if fun == model_place_and_random:
                args.append(rate)
            board = fun(*args)
            if verbose:
                print(board)
            for player in players:
                pl_states[player].append([player_ids[player]] + [i] + list(board.flatten()))
            counter += 1
            winner = evaluate(board)
            #print(f"Winner: {winner}")
            if winner != 0:
                # Winner is either 1, -1 or 0.5
                if verbose:
                    print(board)
                    print("Winner is: " + str(winner))
                for player,states in pl_states.items():
                    i = player_ids[player]
                    for state in states:
                        # If the player wins
                        if i == winner:
                            state.append(1)
                        # If the game is a tie
                        elif winner == 0.5:
                            state.append(0.5)
                        else:
                            state.append(0)
                break
    if gather_data:
        with open('states.txt', 'a') as f:
            states = pl_states["p1"] + pl_states["p2"]
            states = random.sample(states,len(states))
            for state in states:
                f.write(str(state).strip("[]")+"\n")
    # Return the string identifier of the winner
    winner = "p1" if player_ids["p1"] == winner else "p2" if player_ids["p2"] == winner else "tie"
    return (winner)
 

def gather_data():
    print("Gathering data...")
    wins = {"p1":0,"p2":0,"tie":0}
    select_fun = lambda *args: model_place(*args,p=1)
    for i in range(5000):
        winner = play_game(players = {"p1":select_fun,"p2":select_fun}, gather_data = True, shuffle=True, verbose=False)
        wins[winner] += 1
    print(wins)
        
def get_dataset(data, add_channel_dim=False):
    dataset = tf.data.TextLineDataset(data)
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=", "))
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1]), tf.strings.to_number(x[-1])))
    if add_channel_dim:
        dataset = dataset.map(lambda x, y: (tf.expand_dims(x, -1), y))
    return dataset


def simulate_virtual_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    return

def count_num_samples_in_ds(ds):
    """ Count how many y values of 0, 0.5, 1 there are in the dataset.
    """
    num_samples = {0 : 0, 0.5 : 0, 1 : 0}
    for x, y in ds:
        num_samples[y.numpy()] += 1
    return num_samples

def train_model():
    #simulate_virtual_gpus()
    print("Training model...")
    dataset = get_dataset("states.txt",add_channel_dim=True).shuffle(10000)
    num_diff_y_samples = count_num_samples_in_ds(dataset)
    print(num_diff_y_samples)
    print(dataset.take(1).as_numpy_iterator().next())
    #print(dataset.take(1).as_numpy_iterator().next())
    #with tf.device("/GPU:0"):
    if True and os.path.exists("model.h5"):
        model = tf.keras.models.load_model("model.h5")
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(8, 3, activation='relu',input_shape=(11,1)),
            tf.keras.layers.Conv1D(16, 3, activation='relu'),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            #tf.keras.layers.Dense(20, activation='relu'),#,input_shape=(10,)),
            tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                    loss='mse',
                    metrics=['accuracy'])
    
    print(model.summary())
    validation_dataset = dataset.take(500).batch(128)
    test_dataset = dataset.skip(500).take(500).batch(128)
    dataset = dataset.skip(1000).batch(128)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.0001, patience=6, restore_best_weights=True)
    model.fit(dataset, epochs=30,validation_data=validation_dataset, callbacks=[early_stop],verbose=1)
    model.evaluate(test_dataset, verbose=0)
    model.save("model.h5")
    # clear
    tf.keras.backend.clear_session()

def play_games():
    print("Playing games...")
    wins = {"p1":0,"p2":0,"tie":0}
    select_fun = lambda *args: model_place(*args,p=0)
    for i in range(1000):
        winner = play_game(players={"p1":select_fun,"p2":random_place},shuffle=True,verbose=False)
        wins[winner] += 1
    print(wins)
    print("Player 1 won {} times".format(wins["p1"]))
    return wins["p1"]/1000

def play_as_human():
    global MODEL
    MODEL = LiteModel("model.tflite", expand_dims=True)
    winner = play_game(players={"p1":human_place,"p2":model_place},shuffle=True,verbose=True)
    print("Winner: {}".format(winner))

MODEL = None
if __name__ == "__main__":
    #MODEL = LiteModel("model.tflite", expand_dims=True)
    #play_games()
    #play_as_human()
    #exit()
    p1_win_percentage = []
    for i in range(5):
        path = "random" if i == 0 else "model.tflite"
        MODEL = LiteModel(path, expand_dims=True)
        win_perc = play_games()
        p1_win_percentage.append(win_perc)
        if os.path.exists("states.txt"):
            pass
            os.remove("states.txt")
        gather_data()
        if os.path.exists("model.tflite"):
            os.remove("model.tflite")
        train_model()
        convert_to_tflite(file_path="model.h5", output_file="model.tflite")
        print(p1_win_percentage)
    fig,ax = plt.subplots()
    ax.plot(p1_win_percentage, label="Win percentage against random play")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Win percentage")
    ax.set_title("Training progress")
    plt.show()