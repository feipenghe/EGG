# the official cmd
# start_signal_game1.shpython -m egg.zoo.simple_autoenc.train --vocab_size=3 --n_features=6 --n_epoch=50 --max_len=10 --batch_size=512 --random_seed=21

# since I have changed argument, n_features = n_dim * n_properties
python3 -m egg.zoo.simple_autoenc.train --n_objects=4 --n_dim=2 --vocab_size=2 --max_len=5 --sender_cell=rnn --receiver_cell=rnn
# =======
# python3 -m egg.zoo.signal_game.train --root=/Users/pengfeihe/Documents/Programming\ Language/Python/data
# >>>>>>> bb771d9d5754b6b75116676292d59dadb1dbb13e
