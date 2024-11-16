import torch.nn as nn

class ReversibilityClassifier(nn.Module): #classifying neural network structure
    def __init__(self):
        super().__init__()
        input_size = 57
        self.fc1 = nn.Linear(in_features=input_size, out_features=2*input_size)
        self.fc2 = nn.Linear(in_features=2*input_size, out_features=input_size)
        self.fc3 = nn.Linear(in_features=input_size, out_features=1)
        self.reLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.reLU(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.reLU(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def basic_classify(player_pose,player_velocities,predict_reversed,predict_not_reversed):
    """
    :param player_pose: the player's pose
    :param player_velocities: list of player's x and y velocities
    :param predict_reversed: the existing total for predicting reversed
    :param predict_not_reversed: the existing total for predicting not reversed
    :return: Returns the updated lists of predict_reversed and predict_not_reversed
    """
    """
    This is a simple orientation calculation which determines whether the player is facing left/right and up/down
    """
    xleftsum = float(sum(([(player_pose[indx][0]) for indx in range(1, 17) if indx % 2 != 0])))
    xrightsum = float(sum(([(player_pose[indx][0]) for indx in range(1, 17) if indx % 2 == 0])))

    leftvissum = float(sum(([(player_pose[indx][2]) for indx in range(1, 17) if indx % 2 != 0])))
    rightvissum = float(sum(([(player_pose[indx][2]) for indx in range(1, 17) if indx % 2 == 0])))

    x_dir = ""
    y_dir = ""

    if xrightsum > xleftsum:
        y_dir = "up"
    else:
        y_dir = "down"

    if leftvissum > rightvissum:
        x_dir = "left"
    else:
        x_dir = "right"

    if (player_velocities[0] > 0 and x_dir == "right") or (player_velocities[0] < 0 and x_dir == "left"):
        predict_not_reversed += abs(player_velocities[0])

    elif player_velocities[0] != 0 and x_dir != "":
        predict_reversed += abs(player_velocities[0])

    if player_velocities[1] > 0 and y_dir == "down" or player_velocities[1] < 0 and y_dir == "up":
        predict_not_reversed += abs(player_velocities[1])

    elif player_velocities[1] != 0 and y_dir != "":
        predict_reversed += abs(player_velocities[1])

    return predict_reversed, predict_not_reversed

