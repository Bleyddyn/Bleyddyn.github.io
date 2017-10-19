---
title: 'Getting a Keras LSTM layer to work on MaLPi'
date: 2017-10-19
permalink: /posts/2017/10/keras-lstm/
tags:
  - keras
  - lstm
  - malpi
---

When I started trying to use Keras' LSTM layer I realized that the straightforward way to create, train and use a Keras model wasn't going to work. The problem is that the LSTM layer requires either the batch size or the timesteps (e.g. sequence length) to be hard coded in the model. That wasn't going to work when you want to train on large batch sizes and/or sequence lengths, but you need to be able to run one input at a time through the network when it's running on a robot, while maintaining the LSTM's internal state between calls.

I asked about this on [Reddit](https://www.reddit.com/r/MLQuestions/comments/72lzxt/keras_lstm_predict_question/) and [StackOverflow](https://stackoverflow.com/questions/46459843/keras-lstm-predict-1-timestep-at-a-time) without getting completely working answers so I kept experimenting on my own and eventually found a method that I think will work. Having said that, I haven't actually tested this on MaLPi because I don't have enough training data collected, yet.

```python
def make_model_lstm_simple( num_actions, input_dim, batch_size=1, timesteps=None, stateful=False ):
    model = Sequential()

    if stateful:
        input_shape=(batch_size,timesteps) + input_dim
        model.add(TimeDistributed( Convolution2D(16, (8, 8), strides=(4,4), activation='relu' ), batch_input_shape=input_shape, name="Conv-8-16") )
    else:
        input_shape=(timesteps,) + input_dim
        model.add(TimeDistributed( Convolution2D(16, (8, 8), strides=(4,4), activation='relu' ), input_shape=input_shape, name="Conv-8-16") )

    model.add(TimeDistributed( Convolution2D(32, (4, 4), strides=(2,2), activation='relu' ), name="Conv-4-32" ))
    model.add(TimeDistributed( Convolution2D(64, (3, 3), strides=(1,1), activation='relu' ), name="Conv-3-64" ))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True, activation='relu', stateful=stateful ))
    model.add(Dense(num_actions, activation='softmax', name="Output" ))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[metrics.categorical_accuracy] )

    return model

# Training
images, y = loadData()
input_dim = images[0].shape
num_actions = len(y[0])
num_samples = len(images)
timesteps = 10
images = np.reshape( images, (num_samples/timesteps, timesteps) + input_dim )
y = np.reshape( y, (num_samples/timesteps, timesteps, num_actions) )

model = model_keras.make_model_lstm_simple( num_actions, input_dim, stateful=False )
model.fit( images, y, validation_split=0.25, epochs=epochs, shuffle=False )
model.save_weights('model_weights.h5')

# There's probably some way to transfer the weights without writing to file, but for my purposes it doesn't matter much

# Run-time on the robot
model2 = model_keras.make_model_lstm_simple( num_actions, input_dim, batch_size=1, timesteps=1, stateful=True )
model2.load_weights( 'model_weights.h5' )
while True:
    obs = getImageFromCamera()
    obs = np.reshape( obs, (1,1)+input_dim )
    actions = model2.predict( obs, batch_size=1 )
    # choose best action, or e-greedy, etc
    # apply action to robot/world
    # The LSTM layer will retain states between calls unless you call: model2.reset_states()
```

------
